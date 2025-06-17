####### DEBUG SCRIPT TO INVESTIGATE TRAIN VS TEST PERFORMANCE #######
using JLD2, StatsBase, DataFrames, CSV

# Load R² and MSE data for different models and datasets
pooling_types = ["partial_pooling", "no_pooling", "MLE"]
datasets = ["ohashi_low", "ohashi_rich"]

println("="^80)
println("INVESTIGATING TRAIN VS TEST PERFORMANCE")
println("="^80)

performance_summary = DataFrame(
    Model=String[], 
    Dataset=String[], 
    Train_R2_Mean=Float64[], 
    Test_R2_Mean=Float64[],
    Train_R2_Std=Float64[], 
    Test_R2_Std=Float64[],
    Train_MSE_Mean=Float64[], 
    Test_MSE_Mean=Float64[],
    Train_MSE_Std=Float64[], 
    Test_MSE_Std=Float64[],
    R2_Diff=Float64[],  # Test - Train (positive means test is better)
    MSE_Diff=Float64[]  # Train - Test (positive means test is better, i.e., lower MSE)
)

# Check if data files exist and load performance metrics
for model in pooling_types
    for dataset in datasets
        println("\n" * "-"^50)
        println("Checking: $model on $dataset")
        println("-"^50)
        
        # Check R² data
        r2_file = "data/$model/r2_$dataset.jld2"
        mse_file = "data/$model/mse_$dataset.jld2"
        
        if isfile(r2_file)
            try
                r2_data = jldopen(r2_file) do file
                    Dict("train" => file["train"], "test" => file["test"])
                end
                
                train_r2_mean = mean(r2_data["train"])
                test_r2_mean = mean(r2_data["test"])
                train_r2_std = std(r2_data["train"])
                test_r2_std = std(r2_data["test"])
                r2_diff = test_r2_mean - train_r2_mean
                
                println("R² Results:")
                println("  Train: Mean = $(round(train_r2_mean, digits=4)), Std = $(round(train_r2_std, digits=4))")
                println("  Test:  Mean = $(round(test_r2_mean, digits=4)), Std = $(round(test_r2_std, digits=4))")
                println("  Difference (Test - Train): $(round(r2_diff, digits=4))")
                
                if r2_diff > 0.01  # Test significantly better
                    println("  ⚠️  Test R² is notably higher than training R²!")
                end
                
            catch e
                println("  Error loading R² data: $e")
                train_r2_mean = test_r2_mean = train_r2_std = test_r2_std = r2_diff = NaN
            end
        else
            println("  R² file not found: $r2_file")
            train_r2_mean = test_r2_mean = train_r2_std = test_r2_std = r2_diff = NaN
        end
        
        # Check MSE data
        if isfile(mse_file)
            try
                mse_data = jldopen(mse_file) do file
                    file["objectives_current"]  # This is test MSE
                end
                
                test_mse_mean = mean(mse_data)
                test_mse_std = std(mse_data)
                
                println("MSE Results:")
                println("  Test:  Mean = $(round(test_mse_mean, digits=6)), Std = $(round(test_mse_std, digits=6))")
                
                # For MSE, we don't have direct training MSE stored, but we can infer from R²
                train_mse_mean = train_mse_std = mse_diff = NaN
                
            catch e
                println("  Error loading MSE data: $e")
                test_mse_mean = test_mse_std = train_mse_mean = train_mse_std = mse_diff = NaN
            end
        else
            println("  MSE file not found: $mse_file")
            test_mse_mean = test_mse_std = train_mse_mean = train_mse_std = mse_diff = NaN
        end
        
        # Add to summary
        push!(performance_summary, (
            Model=model,
            Dataset=dataset,
            Train_R2_Mean=train_r2_mean,
            Test_R2_Mean=test_r2_mean,
            Train_R2_Std=train_r2_std,
            Test_R2_Std=test_r2_std,
            Train_MSE_Mean=train_mse_mean,
            Test_MSE_Mean=test_mse_mean,
            Train_MSE_Std=train_mse_std,
            Test_MSE_Std=test_mse_std,
            R2_Diff=r2_diff,
            MSE_Diff=mse_diff
        ))
    end
end

println("\n" * "="^80)
println("PERFORMANCE SUMMARY TABLE")
println("="^80)
show(performance_summary, allrows=true, allcols=true)

# Identify cases where test performance is suspiciously better
println("\n" * "="^80)
println("SUSPICIOUS CASES (Test significantly better than Train)")
println("="^80)

suspicious_cases = filter(row -> !isnan(row.R2_Diff) && row.R2_Diff > 0.02, performance_summary)
if nrow(suspicious_cases) > 0
    show(suspicious_cases, allrows=true, allcols=true)
    
    println("\n⚠️  POTENTIAL ISSUES DETECTED:")
    for row in eachrow(suspicious_cases)
        println("  - $(row.Model) on $(row.Dataset): Test R² is $(round(row.R2_Diff, digits=4)) higher than train R²")
    end
else
    println("No suspicious cases detected.")
end

# Save results
CSV.write("data/train_vs_test_performance_debug.csv", performance_summary)
println("\nDebug results saved to: data/train_vs_test_performance_debug.csv")

# Also check if there are any data leakage issues by examining the data splits
println("\n" * "="^80)
println("CHECKING DATA SPLITS FOR POTENTIAL LEAKAGE")
println("="^80)

for dataset in datasets
    data_file = "data/$dataset.jld2"
    if isfile(data_file)
        try
            train_data, test_data = jldopen(data_file) do file
                file["train"], file["test"]
            end
            
            println("\nDataset: $dataset")
            println("  Train subjects: $(size(train_data.cpeptide, 1))")
            println("  Test subjects: $(size(test_data.cpeptide, 1))")
            println("  Train subject indices: $(train_data.training_indices[1:min(10, length(train_data.training_indices))])...")
            
            # Check for any overlap in subject IDs if they exist
            if haskey(train_data, :subject_ids) && haskey(test_data, :subject_ids)
                overlap = intersect(train_data.subject_ids, test_data.subject_ids)
                if !isempty(overlap)
                    println("  ⚠️  OVERLAP DETECTED: $(length(overlap)) subjects appear in both train and test!")
                    println("  Overlapping subjects: $overlap")
                else
                    println("  ✓ No subject overlap detected")
                end
            else
                println("  Subject IDs not available for overlap check")
            end
            
        catch e
            println("  Error loading data: $e")
        end
    else
        println("  Data file not found: $data_file")
    end
end

println("\n" * "="^80)
println("DEBUG ANALYSIS COMPLETE")
println("="^80)
