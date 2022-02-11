# Author: Jeremy Boyd (jeremyboyd@pm.me)
# Description: Fit, check, & visualize Bayesian hierarchical models of vehicle
# cutting data.

# Resources
library(tidyverse)
library(brms)
library(tidybayes)
library(feather)
library(lubridate)
library(tictoc)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#### Read in & organize data ####
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

df <- read_feather("cutting_data.feather") %>%

    # Drop rows with missing values in any of the following columns
    filter(if_all(.cols = c("cut", "traffic", "driver_sex", "driver_age"),
                  ~ !is.na(.))) %>%
    
    # Convert hour:minute time to hours
    mutate(hour = hour(datetime),
           minute = minute(datetime),
           time_hours = hour + (minute / 60),
           
           # Recode driver sex & traffic as dummies indicating male sex and
           # heavy traffic, respectively.
           male = if_else(driver_sex == "m", 1, 0),
           traffic_heavy = if_else(traffic > 2, 1, 0),
           
           # Center most predictors; center & scale time_hours
           c_vehicle_status = vehicle_status - mean(vehicle_status),
           c_traffic_heavy = traffic_heavy - mean(traffic_heavy),
           c_male = male - mean(male),
           c_driver_age = driver_age - mean(driver_age),
           c_time_hours = scale(time_hours))

# Write data to feather
write_feather(df, "cutting_data_organized.feather")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#### Explore priors ####
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Cutting priors. Alpha represents the log odds of a cut when all other
# predictors are at zero: vehicle status, traffic, driver age, driver sex.
# Status, traffic, and age are all centered at their means: for status it's
# 3.03, for traffic it's 2.46, for age it's 1.57, which is close to the 2
# (middle) category. For driver sex, centering means that zero is balanced in
# between male and female.

# Assume that cutting happens less than half the time. This gives the log odds
# of cutting 25% of the time:
qlogis(.25)

# Could model the log odds of cutting as normally distributed with mean of -1
# and sd of 4. This ends up favoring extreme values. But normal(-1, 1) looks
# pretty good. Density with the most mass is around 13-14%, which seems
# reasonable. Has long tail going to 1, so allows higher values as well.
samples_logodds <- tibble(alpha = rnorm(100000, -1, 1))
samples_prob <- tibble(p = plogis(samples_logodds$alpha))
ggplot(data = samples_logodds, mapping = aes(alpha)) + geom_density()
ggplot(data = samples_prob, mapping = aes(p)) + geom_density()

# Function that makes predictions in a logistic regression model of cutting.
# Takes vectors of alpha samples, beta samples, and vehicle status values as
# input and uses them to output cutting predictions (1/0).
logistic_model_pred <- function(alpha_samples,
                                beta_samples,
                                vehicle_status,
                                N_obs) {
    map2_dfr(alpha_samples, beta_samples, function(alpha, beta) {
        tibble(
            vehicle_status = vehicle_status,
            c_vehicle_status = vehicle_status - mean(vehicle_status),
            
            # Compute likelihood. Convert to probability using link function.
            theta = plogis(alpha + c_vehicle_status * beta),
            cut = rbernoulli(N_obs, p = theta)
        )
    },
    .id = "iter") %>%
        
        # .id is always a string and needs to be converted to a number
        mutate(iter = as.numeric(iter))
}

# Loop over candidate beta SDs to create beta samples from different
# distributions.
N_obs <- 1000
vehicle_status <- rep(c(1, 2, 3, 4, 5), 200)
alpha_samples <- rnorm(1000, -1, 1)
sds_beta <- c(1, 0.5, 0.1, 0.01, 0.001)
prior_pred <- map_dfr(sds_beta, function(sd) {
    beta_samples <- rnorm(1000, 0, sd)
    logistic_model_pred(
        alpha_samples = alpha_samples,
        beta_samples = beta_samples,
        vehicle_status = vehicle_status,
        N_obs = N_obs
    ) %>%
        mutate(prior_beta_sd = sd)
})

# Compute mean proportion of cuts for each combination of prior_beta_sd, iter, &
# vehicle_status.
mean_cut <-prior_pred %>%
    group_by(prior_beta_sd, iter, vehicle_status) %>%
    summarize(p_cut = mean(cut), .groups = "drop") %>%
    mutate(prior = paste0("Normal(0, ", prior_beta_sd, ")"))

# Visualize the resulting distribution of cutting probabilities. They all look
# pretty reasonable, with most of the mass around the lower probabilities. I'll
# use normal(0, 1) just because that's simplest.
mean_cut %>%
    ggplot(aes(p_cut)) +
    geom_histogram() +
    facet_grid(vehicle_status ~ prior)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#### Fit model, check rhats, neff ratios ####
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Custom priors based on the exploration above
cut_priors <- c(
    
    # We expect that cutting isn't super likely. This starts with a probability
    # ~ 12-13%, with a long tail to the right.
    prior(normal(-1, 1), class = Intercept),
    
    # Used as the prior for all class b (betas) based on the exploration in the
    # previous section.
    prior(normal(0, 1), class = b),
    
    # Used for all varying effects SDs
    prior(cauchy(0, 1), class = sd),
    
    # We don't expect super strong correlations
    prior(lkj(2), class = cor)
)

# Fit hierarchical logistic regression model with full random effects structure
tic()
fit_cut <- brm(
    cut ~ 1 + c_vehicle_status * c_traffic_heavy + c_male + c_driver_age +
            c_time_hours +
        (1 + (c_vehicle_status * c_traffic_heavy + c_male + c_driver_age +
                  c_time_hours)|coder_id) +
        (1 + (c_vehicle_status * c_traffic_heavy + c_male + c_driver_age +
                  c_time_hours)|inter_id),
    data = df,
    prior = cut_priors,
    family = bernoulli(link = logit),
    chains = 4,
    iter = 2000,
    warmup = 1000,
    cores = 4,
    seed = 1,
    
    # The cmdstanr backend should be faster, but I'm using the default rstan
    # bakcend instead because when the model is fit with cmdstanr I get some
    # warnings I don't like when running mcmc_plot().
    # backend = "cmdstanr"
    )
toc()

# Save model for later use
if(!file.exists("fit_cut.RDS")) {
    saveRDS(fit_cut, "fit_cut.RDS")
}
 
# Check priors
prior_summary(fit_cut)

# All rhats <= 1.05
mcmc_plot(fit_cut, type = "rhat_hist")

# Some neff ratios <= 0.5, but > 0.1.
mcmc_plot(fit_cut, type = "neff_hist")

# Parameters with low neff
low_neff <- neff_ratio(fit_cut)[neff_ratio(fit_cut) < 0.5]

# Manually inspect traceplots of parameters with low neff. All look well mixed.
plot(fit_cut, variable = names(low_neff))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#### Posterior predictive checks ####
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Samples from posterior are nicely matching empirical density
pp_check(fit_cut, ndraws = 100, type = "dens_overlay")

# Shows counts of 0 & 1 (no cut/cut) and whether they're near empirical data
pp_check(fit_cut, ndraws = 100, type = "bars")

# Table relating vehicle status values to their centered versions
status_tbl <- df %>%
    select(vehicle_status) %>%
    unique() %>%
    arrange(vehicle_status) %>%
    mutate(c_vehicle_status = vehicle_status - mean(df$vehicle_status))

# Assign labels to each centered level of vehicle status
status_labels <- paste("Vehicle status = ", 1:5) %>%
    setNames(status_tbl$c_vehicle_status)

# Counts of 0/1 predictions (no cut/cut) for each level of vehicle status.
# Indicates that the model is good at representing the data. Setting ndraws to
# NULL computes yrep from all draws--in this case 4K--which is the most granular
# level of detail available regarding the posterior.
pp_check(fit_cut,
         ndraws = NULL,
         type = "bars_grouped",
         group = "c_vehicle_status",
         facet_args = list(labeller = as_labeller(status_labels))) +
    scale_x_continuous(name = "Cut", breaks = c(0, 1)) +
    labs(title = "Predicted distribution of cuts by vehicle status",
         subtitle = "Based on 4K replications")

# Table relating traffic values to their centered versions
traffic_tbl <- df %>%
    select(traffic_heavy) %>%
    unique() %>%
    arrange(traffic_heavy) %>%
    mutate(c_traffic_heavy = traffic_heavy - mean(df$traffic_heavy))

# Assign labels to each centered level of traffic
traffic_labels <- paste("Traffic = ", c("Light", "Heavy")) %>%
    setNames(traffic_tbl$c_traffic_heavy)

# Counts of 0/1 for each level of traffic
pp_check(fit_cut,
         ndraws = NULL,
         type = "bars_grouped",
         group = "c_traffic_heavy",
         facet_args = list(labeller = as_labeller(traffic_labels))) +
    scale_x_continuous(name = "Cut", breaks = c(0, 1)) +
    labs(title = "Predicted distribution of cuts by traffic",
         subtitle = "Based on 4K replications")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#### Visualize status x traffic interaction ####
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Table of conditions to make predictions for. Note that male, driver age, and
# time hours are set to their centered values of 0.
new_data <- expand_grid(
    c_vehicle_status = unique(df$c_vehicle_status),
    c_traffic_heavy = unique(df$c_traffic_heavy),
    c_male = 0,
    c_driver_age = 0,
    c_time_hours = 0)
# new_data$cond_num <- 1:nrow(new_data)

# Generate log-odds predictions for new data. We're mainly interested in the
# fixed effects of vehicle status, traffic, and their interaction, so use
# re_formula = NA to ignore random effect adjustments.
pred_lodds <- linpred_draws(
    object = fit_cut,
    newdata = new_data,
    re_formula = NA)

# Summarize log-odds predictions across all samples for each condition, then
# convert to probabilities.
pred_probs <- pred_lodds %>%
    ungroup() %>%
    select(c_vehicle_status, c_traffic_heavy, .linpred) %>%
    group_by(c_vehicle_status, c_traffic_heavy) %>%
    mean_hdi(.width = 0.80) %>%
    mutate(across(.cols = all_of(c(".linpred", ".lower", ".upper")),
                  plogis)) %>%
    rename(mean = .linpred) %>%
    left_join(status_tbl, by = "c_vehicle_status") %>%
    left_join(traffic_tbl, by = "c_traffic_heavy") %>%
    mutate(Traffic = if_else(traffic_heavy == 1, "Heavy", "Light"))

# Visualize interaction. Exactly matches what you get when you plot using
# conditional_effects(). Plotting 80% CIs makes it clear that there's
# potentially an interaction effect. But it's actually in the opposite direction
# from what we predicted for the attention account. Instead of a stronger effect
# of status under heavy traffic conditions, we get a weaker effect. Could be
# consistent with heavy traffic being confusing for all drivers, which bumps up
# the base rate of cutting for lower status drivers, which attenuates the status
# effect. Another way of thinking about this is that there's a ceiling on the
# amount of cutting that a driver does. High status drivers are at the ceiling,
# so changing traffic doesn't affect them as munch. But low status drivers
# aren't, so increasing traffic increases the probability that they'll cut.
theme_set(theme_minimal())
pred_probs %>%
    ggplot(mapping = aes(x = vehicle_status,
                         y = mean,
                         group = Traffic)) +
    geom_ribbon(mapping = aes(ymin = .lower,
                                  ymax = .upper,
                                  fill = Traffic),
                    alpha = 0.2) +
    geom_line(aes(color = Traffic)) +
    scale_x_continuous(name = "Vehicle Status") +
    scale_y_continuous(name = "P(Cut)", limits = c(0, 1)) +
    scale_color_brewer(type = "qual", palette = "Set1") +
    scale_fill_brewer(type = "qual", palette = "Set1") +
    labs(
        title = "Probability of cutting as a function of vehicle status and traffic",
        subtitle = "Lines represent predicted means. Shading shows 80% credible intervals.") +
    theme(axis.title.y = element_text(angle = 0, vjust = 0.5),
          panel.grid.major = element_line(color = "gray90", size = .2))

# The interaction is credible at 80%. Highest level at which it's still credible
# is 88%. So we have fairly good evidence that the vehicle status effect isn't
# the same in light versus heavy traffic.
posterior_interval(fit_cut,
                   variable = "b_c_vehicle_status:c_traffic_heavy",
                   prob = 0.80)
posterior_interval(fit_cut,
                   variable = "b_c_vehicle_status:c_traffic_heavy",
                   prob = 0.88)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#### Is there a status effect for light traffic only? ####
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Light traffic only
light <- df %>%
    filter(traffic_heavy == 0) %>%
    
    # Re-center variables; rescale time
    mutate(c_vehicle_status = vehicle_status - mean(vehicle_status),
           c_male = male - mean(male),
           c_driver_age = driver_age - mean(driver_age),
           c_time_hours = scale(time_hours))
    
# Fit model
tic()
fit_light <- brm(
    cut ~ 1 + c_vehicle_status + c_male + c_driver_age + c_time_hours +
        (1 + (c_vehicle_status + c_male + c_driver_age +
                  c_time_hours)|coder_id) +
        (1 + (c_vehicle_status + c_male + c_driver_age +
                  c_time_hours)|inter_id),
    data = light,
    prior = cut_priors,
    family = bernoulli(link = logit),
    chains = 4,
    iter = 2000,
    warmup = 1000,
    cores = 4,
    seed = 1)
toc()

# Save model for later use
if(!file.exists("fit_light.RDS")) {
    saveRDS(fit_light, "fit_light.RDS")
}

# All rhats <= 1.05
mcmc_plot(fit_light, type = "rhat_hist")

# Some neff ratios <= 0.5, but > 0.1.
mcmc_plot(fit_light, type = "neff_hist")

# 58 total parameters have neff ratios <= 0.5 but > 0.1
low_neff_light <- neff_ratio(fit_light)[neff_ratio(fit_light) < 0.5]

# Manually inspect traceplots of parameters with low neff. All look well mixed.
plot(fit_light, variable = names(low_neff_light))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#### Is there a status effect for heavy traffic only? ####
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Heavy traffic only
heavy <- df %>%
    filter(traffic_heavy == 1) %>%
    
    # Re-center variables; rescale time
    mutate(c_vehicle_status = vehicle_status - mean(vehicle_status),
           c_male = male - mean(male),
           c_driver_age = driver_age - mean(driver_age),
           c_time_hours = scale(time_hours))

# Fit model
tic()
fit_heavy <- brm(
    cut ~ 1 + c_vehicle_status + c_male + c_driver_age + c_time_hours +
        (1 + (c_vehicle_status + c_male + c_driver_age +
                  c_time_hours)|coder_id) +
        (1 + (c_vehicle_status + c_male + c_driver_age +
                  c_time_hours)|inter_id),
    data = heavy,
    prior = cut_priors,
    family = bernoulli(link = logit),
    chains = 4,
    iter = 2000,
    warmup = 1000,
    cores = 4,
    seed = 1)
toc()

# Save model for later use
if(!file.exists("fit_heavy.RDS")) {
    saveRDS(fit_heavy, "fit_heavy.RDS")
    }

# All rhats <= 1.05
mcmc_plot(fit_heavy, type = "rhat_hist")

# Some neff ratios <= 0.5, but > 0.1.
mcmc_plot(fit_heavy, type = "neff_hist")

# 12 total parameters have neff ratios <= 0.5 but > 0.1
low_neff_heavy <- neff_ratio(fit_heavy)[neff_ratio(fit_heavy) < 0.5]

# Manually inspect traceplots of parameters with low neff. All look well mixed.
plot(fit_heavy, variable = names(low_neff_heavy))

# This is evidence that the interaction is credible: the vehicle status effect
# is not credible at theh 95% level when trafic is heavy, but it is credible at
# the 95% level when traffic is light.
posterior_interval(fit_heavy,
                   variable = "b_c_vehicle_status",
                   prob = 0.95)
posterior_interval(fit_light,
                   variable = "b_c_vehicle_status",
                   prob = 0.95)
