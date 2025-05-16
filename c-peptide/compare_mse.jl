using JLD2
using CSV
using DataFrames

# Load MSE
mse_mle = JLD2.load("data/mle/mse.jld2")
mse_pp = JLD2.load("data/partial_pooling/mse.jld2")
mse_mle = mse_mle["test"]
mse_pp = mse_pp["objectives_current"]
# MSE per type: T2DM, NGT, IGT
types = ["T2DM", "NGT", "IGT"]

# Load the data to obtain indices for the types
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

results = DataFrame(Type=String[], MLE_CI_=Float64[], MLE_CI_Upper=Float64[], PP_CI_Lower=Float64[], PP_CI_Upper=Float64[], t_statistic=Float64[], p_value=Float64[], significant=String[], n=Int[])

for (i, type) in enumerate(unique(test_data.types))
    # Get the MSE values for this type
    type_indices = test_data.types .== type
    type_mse_mle = mse_mle[type_indices]
    type_mse_pp = mse_pp[type_indices]

    # get mean and std
    mu_mse_mle = mean(type_mse_mle)
    mu_mse_pp = mean(type_mse_pp)
    std_mse_mle = std(type_mse_mle)
    std_mse_pp = std(type_mse_pp)

    # Calculate the 99% CI
    alpha = 0.01
    lb_mse_mle = mu_mse_mle - quantile(Normal(0, 1), 1 - alpha / 2) * (std_mse_mle / sqrt(length(type_mse_mle)))
    ub_mse_mle = mu_mse_mle + quantile(Normal(0, 1), 1 - alpha / 2) * (std_mse_mle / sqrt(length(type_mse_mle)))
    lb_mse_pp = mu_mse_pp - quantile(Normal(0, 1), 1 - alpha / 2) * (std_mse_pp / sqrt(length(type_mse_pp)))
    ub_mse_pp = mu_mse_pp + quantile(Normal(0, 1), 1 - alpha / 2) * (std_mse_pp / sqrt(length(type_mse_pp)))

    println("Type: $type")
    println("MLE 99% CI: ($lb_mse_mle, $ub_mse_mle)")
    println("Partial Pooling 99% CI: ($lb_mse_pp, $ub_mse_pp)")

    # Perform a t-test
    t_statistic = (mu_mse_mle - mu_mse_pp) / sqrt((std_mse_mle^2 / length(type_mse_mle)) + (std_mse_pp^2 / length(type_mse_pp)))
    df = length(type_mse_mle) + length(type_mse_pp) - 2
    p_value = 2 * (1 - cdf(TDist(df), abs(t_statistic)))
    println("t-statistic: $t_statistic")
    println("p-value: $p_value")
    if p_value < 0.01
        println("The difference is statistically significant.")
    else
        println("The difference is not statistically significant.")
    end
    println("n: ", length(type_mse_mle) + length(type_mse_pp))
    println("--------------------------------------------------")
    significant = p_value < 0.01 ? "True" : "False"
    
    # Round the results to 3 decimal places
    digits = 4
    lb_mse_mle = round(lb_mse_mle, digits=digits)
    ub_mse_mle = round(ub_mse_mle, digits = digits)
    lb_mse_pp = round(lb_mse_pp, digits= digits)
    ub_mse_pp = round(ub_mse_pp, digits= digits)
    t_statistic = round(t_statistic, digits = digits)
    p_value = round(p_value, digits= digits)

    # Append the results to the DataFrame
    push!(results, (string(type), lb_mse_mle, ub_mse_mle, lb_mse_pp, ub_mse_pp, t_statistic, p_value, significant, length(type_mse_mle) + length(type_mse_pp)))
end

CSV.write("data/compare_mse_results.csv", results)
