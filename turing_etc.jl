using Turing, DataFrames, Random, Distributions, StatsPlots, MCMCChains, StatsBase

# suppose we want to model blood glucose
# set a model

@model function estimate_glucose(glu)
    mu ~ Normal(7,2)
    sigma ~ Exponential(1)
    for i in 1:length(glu)
        glu ~ Normal(mu, sigma)
    end

end

# get some dummy data
glucose_data = [6,7,5,6,7,4,5,6,7]

# set up the chain
chain = sample(estimate_glucose(glucose_data), NUTS(), 1_000)

# get summarystats
summary_stats = summarystats(chain)

# mean for glucose 5.89, sigma for glucose 1.81, ess = 643, ess for sigma: 538, rhat 1.00, 0.99
sumdf = DataFrame(summary_stats)

# Extract the data frame
chainsdf = DataFrame(chain)

# plot the chain
plot(chain)
# not quite the fuzzy caterpillar but workable

# Do predictions
# create empty data 
test_data = []
# set up Prior test chain
prior_chain = sample(estimate_glucose(test_data), Prior(), 1_000)

# Run predictions
# Prior predictions
prior_prediction = predict(estimate_glucose(test_data), prior_chain)
# Inspect
prior_predict = DataFrame(prior_prediction)
# Posterior Prediction
posterior_prediction = predict(estimate_glucose(test_data), chain)
posterior_predict = DataFrame(posterior_prediction)

## --- Linear Regression
# read glucose data
using CSV
# Download url data for glucose data
url = "https://raw.githubusercontent.com/arinbasu/mcmc_learing/refs/heads/main/diabetes.csv"
# create a data frame from the glucose data
df = CSV.read(download(url), DataFrame)

# For this demo, we will use a trivial example, where we will analyse the relationship between BMI and Glucose
# We hypothesise the Glucose and BMI are Normally distributed, and that
# glucose ~ alpha + b * BMI 
# here alpha is the intercept and has a normal distribution, and b is the slope also normally distributed
# Plot glucose
histogram(df.Glucose, legend=false, title="Glucose Data")
# remove the 0 data poiht
df1 = subset(df, :Glucose => ByRow(x -> x > 0))
# Now do the plot again and confirm
histogram(df1.Glucose, legend=false, title="Glucose Data")

# set up the model
@model function regress_glucose(glu, bmi)
    alpha ~ Normal(25,4)
    beta ~ Normal(0,1)
    sigma ~ Exponential(1)
    mu = alpha .+ beta .* (bmi .- 25)
    glu ~ MvNormal(mu, sigma)
    
end

# In the model we have tried to model how does beta fare
# for each unit of BMI above 25 units as that's the limit of overweight
# sample the posterior distribution
chainglu = sample(regress_glucose(df1.Glucose, df1.BMI), NUTS(), 1_000)
# extract the dataframe
chaingludf = DataFrame(chainglu)

# let's examine the summarystats
summary_glu = DataFrame(summarystats(chainglu))
# we see that ESS is around 480 and above, rhat is about 1.00
# is the TracePlot like fuzzy caterpillar?
plot(chainglu)
# Not quite and that tells us that this model is not a good model although it converged

