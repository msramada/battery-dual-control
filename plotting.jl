using JLD2
using Statistics
using LaTeXStrings
using Plots, LinearAlgebra
@load "simulation_results100.jld2" x_rec u_rec cov_rec x_true_rec cost_rec est_err_rec x_rec_mpc u_rec_mpc cov_rec_mpc x_true_rec_mpc cost_rec_mpc est_err_rec_mpc

### Grab required parameters from main.jl
num_simulations = length(x_rec)
T = length(x_rec[1])
n = length(x_rec[1][1])
set_point = 1.0
Q = 1.0
R = 0.1

function running_cost(x, u)
    cost = Q * (sum(x) - set_point) ^ 2  + u' * R * u
    # penalize the difference between SOCs
    for i in 1:n
        for j in i+1:n
            cost += Q * (x[i] - x[j])^2
        end
    end
    return cost
end

nonlinearMPC_cost = [sum(running_cost(x_true_rec[i][k],u_rec[i][k])/T for k in 1:T) for i in 1:num_simulations]
linearMPC_cost = [sum(running_cost(x_true_rec_mpc[i][k],u_rec_mpc[i][k])/T for k in 1:T) for i in 1:num_simulations]

nonlinearMPC_cov = [sum(tr(cov_rec[i][k]) for k in 1:T)/T for i in 1:num_simulations]
linearMPC_cov = [sum(tr(cov_rec_mpc[i][k]) for k in 1:T)/T for i in 1:num_simulations]

function running_cost_stochastic(info_state, u)
    x = info_state[1:n]
    Σ = info_state[n+1:end]
    Σ = reshape(Σ, n, n)
    cost = running_cost(x, u) + Q * tr(Σ)
    # penalize the difference between SOCs
    for i in 1:n
        for j in i+1:n
            cost += Q * (Σ[i, i] - 2 * Σ[i, j] + Σ[j, j])
        end
    end
    # cost += Q*sum((Σ .- Σ').^2)/2
    return cost
end



algo_name = "Algorithm 1"
nominal_name = "Linear MPC"
fontSize=13
### Use to plot histograms of cost and estimation error to compare both methods
function histograms()
    # # Create a 2x2 layout for the subplots
    # nonlinearMPC_cost = [sum(running_cost(x_true_rec[i][k],u_rec[i][k]) for k in 1:T) for i in 1:num_simulations]
    # linearMPC_cost = [sum(running_cost(x_true_rec_mpc[i][k],u_rec_mpc[i][k]) for k in 1:T) for i in 1:num_simulations]

    # nonlinearMPC_cov = [sum(tr(cov_rec[i][k]) for k in 1:T)/T for i in 1:num_simulations]
    # linearMPC_cov = [sum(tr(cov_rec_mpc[i][k]) for k in 1:T)/T for i in 1:num_simulations]

    # Calculate means
    mean_nonlinearMPC_cov = sum(est_err_rec)/num_simulations
    mean_linearMPC_cov = sum(est_err_rec_mpc)/num_simulations
    mean_nonlinearMPC_cost = sum(nonlinearMPC_cost)/num_simulations
    mean_linearMPC_cost = sum(linearMPC_cost)/num_simulations

    plot_layout = @layout [a b; c d]


    # Define consistent bin edges for estimation error histograms
    num_bins = 21
    min_estimation = minimum([minimum(est_err_rec), minimum(est_err_rec_mpc)])
    max_estimation = maximum([maximum(est_err_rec), maximum(est_err_rec_mpc)])
    bins_estimation = range(min_estimation, (max_estimation + (max_estimation - min_estimation) / (num_bins - 1)), length=num_bins)


    # Define consistent bin edges for cost histograms
    min_cost = minimum([minimum(nonlinearMPC_cost), minimum(linearMPC_cost)])
    max_cost = maximum([maximum(nonlinearMPC_cost), maximum(linearMPC_cost)])
    bins_cost = range(min_cost, max_cost + (max_cost - min_cost) / (num_bins - 1), length=num_bins)  # Add one extra bin to the right

    # Individual histograms with vertical dashed lines for means
    p1 = histogram(est_err_rec, 
        bins=bins_estimation, 
        label=algo_name, 
        color=:blue, 
        xlabel="Estimation Error", 
        ylabel="Frequency",
        legend=:topright)
    vline!(p1, [mean_nonlinearMPC_cov], label="Mean", color=:red, linestyle=:dash, linewidth=1.5)
    ylims!(0.0,30)

    p2 = histogram(est_err_rec_mpc, 
        bins=bins_estimation, 
        label=nominal_name, 
        color=:green, 
        xlabel="Estimation Error", 
        ylabel="Frequency",
        legend=:topright)
    vline!(p2, [mean_linearMPC_cov], label="Mean", color=:red, linestyle=:dash, linewidth=1.5)
    ylims!(0.0,30)


    p3 = histogram(nonlinearMPC_cost, 
        bins=bins_cost, 
        label=algo_name, 
        color=:blue, 
        xlabel="Cost", 
        ylabel="Frequency")
    vline!(p3, [mean_nonlinearMPC_cost], label="Mean", color=:red, linestyle=:dash,linewidth=1.5)
    ylims!(0.0,30)

    p4 = histogram(linearMPC_cost, 
        bins=bins_cost, 
        label=nominal_name, 
        color=:green, 
        xlabel="Cost", 
        ylabel="Frequency")
    vline!(p4, [mean_linearMPC_cost], label="Mean", color=:red, linestyle=:dash,linewidth=1.5)
    ylims!(0.0,30)

    plot(p2, p4, p1, p3,
        layout=plot_layout, 
        legend=:topright,
        legendfontsize=12)

    plot!(
        framestyle = :box,
        yguidefontsize = fontSize,
        xguidefontsize = fontSize,
        xlabelfontsize = fontSize,
        ylabelfontsize = fontSize,
        xtickfontsize = 8,
        ytickfontsize = 8,
        palette = :seaborn_muted,
        foreground_color_legend = nothing,
        legendfontsize=fontSize-1,
        fontfamily = "Computer Modern");

    savefig("./histograms.png")

    # plot(p2,p4,
    #     layout=plot_layout, 
    #     legend=:topright,
    #     legendfontsize=7)

    # savefig("./MPC_histograms.png")

end

function plot_OCV_SOC()
    # Define the SOC range
    SOC_range = 0:0.01:1  # SOC values from 0 to 1 with a step of 0.01

    # Compute OCV values for each SOC in the range
    OCV_values = [measurement_dynamics([SOC, SOC, SOC]) for SOC in SOC_range]
    OCV_values = reduce(hcat, OCV_values)  # Combine into a matrix for easier plotting

    # Plot each OCV-SOC curve
    plot(
        SOC_range, OCV_values[1, :], 
        label="Lithium-Ion", 
        linewidth=2, 
        linestyle=:solid, 
        color=:blue
    )
    plot!(
        SOC_range, OCV_values[2, :], 
        label="Lithium Titanate Oxide", 
        linewidth=2, 
        linestyle=:dash, 
        color=:red
    )
    plot!(
        SOC_range, OCV_values[3, :], 
        label="Lithium Sulfur",
        linewidth=2, 
        linestyle=:dot, 
        color=:green,
        size=(650,250),
        legendfontsize=11,
        left_margin=5Plots.mm,
        bottom_margin=5Plots.mm,
        top_margin=5Plots.mm,
        legend_position=(0.15,0.85)
    )

    xlabel!("SOC")
    ylabel!("OCV (up to additive constant)")
    # Save the plot

    plot!(
        framestyle = :box,
        yguidefontsize = fontSize,
        xguidefontsize = fontSize,
        xtickfontsize = 8,
        ytickfontsize = 8,
        xlabelfontsize = fontSize,
        ylabelfontsize = fontSize,
        palette = :seaborn_muted,
        foreground_color_legend = nothing,
        legendfontsize=fontSize-1,
        fontfamily = "Computer Modern",
        xlabel = "SOC",
        ylabel = "OCV",);
    savefig("./OCV.png")
end


function simulation_averaged_sum_of_states()
    # Initialize vectors to store the average sum of states for each time step
    avg_sum_states = zeros(T)
    avg_sum_states_mpc = zeros(T)

    # Loop over all simulations and compute the sum of states for each time step
    for i in 1:num_simulations
        for t in 1:T
            # Accumulate the sum of states
            avg_sum_states[t] += sum(x_true_rec[i][t])
            avg_sum_states_mpc[t] += sum(x_true_rec_mpc[i][t])
        end
    end

    # Divide by the number of simulations to get the average
    avg_sum_states ./= num_simulations
    avg_sum_states_mpc ./= num_simulations

    # Plot the average sum of states across time
    plot(1:T, avg_sum_states, 
        xlabel="Time Step", 
        ylabel="Average Sum of States", 
        label="Stochastic Optimal Control", 
        color=:blue)
    plot!(1:T, avg_sum_states_mpc, 
        label="Model Predictive Control", 
        color=:green)

    savefig("./average_sum_of_states.png")
end

function compare_covariance_trace()

    # nonlinearMPC_cov = [sum(tr(cov_rec[i][k]) for k in 1:T) for i in 1:num_simulations ] / (num_simulations * T)
    # linearMPC_cov = [sum(tr(cov_rec_mpc[i][k]) for k in 1:T) for i in 1:num_simulations ] / (num_simulations * T)

    # Calculate the min and max for each dataset
    min_nonlinear = minimum(nonlinearMPC_cov)
    max_nonlinear = maximum(nonlinearMPC_cov)
    min_linear = minimum(linearMPC_cov)
    max_linear = maximum(linearMPC_cov)

    # Calculate the range (difference between max and min)
    range_nonlinear = max_nonlinear - min_nonlinear
    range_linear = max_linear - min_linear

    # Determine the larger range
    max_range = max(range_nonlinear, range_linear)

    # Adjust the smaller range to match the larger one
    if range_nonlinear < max_range
        max_nonlinear = min_nonlinear + max_range
    elseif range_linear < max_range
        max_linear = min_linear + max_range
    end

    # Define consistent bin edges for both histograms
    bins_nonlinear = range(min_nonlinear, max_nonlinear, length=41)
    bins_linear = range(min_linear, max_linear, length=41)

    p1 = histogram(
        nonlinearMPC_cov, 
        bins=bins_nonlinear, 
        label=algo_name, 
        color=:blue, 
        xlabel="tr(Σ)", 
        ylabel="Frequency",
        legend=:topright)
    vline!(p1, [sum(nonlinearMPC_cov)/num_simulations], label="Mean", color=:red, linestyle=:dash,linewidth=1.5)

    p2 = histogram(
        linearMPC_cov, 
        bins=bins_linear, 
        label=nominal_name, 
        color=:green, 
        xlabel="tr(Σ)", 
        ylabel="Frequency",
        legend=:topright)
    vline!(p2, [sum(linearMPC_cov)/num_simulations], label="Mean", color=:red, linestyle=:dash,linewidth=1.5)


    # plot_layout = @layout [a, b]

    plot(p1, p2,
        layout=(1,2),
        size=(650,250),
        left_margin=5Plots.mm,
        bottom_margin=5Plots.mm,
        top_margin=5Plots.mm,
        legendfontsize=fontSize-1)

    ylims!(0.0, 25)


    plot!(
    framestyle = :box,
    yguidefontsize = fontSize,
    xguidefontsize = fontSize,
    xlabelfontsize = fontSize,
    ylabelfontsize = fontSize,
    xtickfontsize = 8,
    ytickfontsize = 8,
    palette = :seaborn_muted,
    foreground_color_legend = nothing,
    legendfontsize=fontSize-1,
    fontfamily = "Computer Modern");
    
    savefig("./covariance_trace.png")


    nonlinearMPC_avg_trace_cov = sum(nonlinearMPC_cov) / (num_simulations)
    linearMPC_avg_trace_cov = sum(linearMPC_cov) / (num_simulations)

    println("Average covariance trace nonlinear MPC: ", nonlinearMPC_avg_trace_cov)
    println("Average covariance trace linear MPC: ", linearMPC_avg_trace_cov)
    println("Average covariance trace change: % ", (nonlinearMPC_avg_trace_cov - linearMPC_avg_trace_cov) / linearMPC_avg_trace_cov * 100)

end

function compare_costs()

    # info_state_rec = [[x_rec[i][k];vec(cov_rec[i][k])] for i in 1:num_simulations, k in 1:T]
    # info_state_rec_mpc = [[x_rec_mpc[i][k];vec(cov_rec_mpc[i][k])] for i in 1:num_simulations, k in 1:T]


    # nonlinearMPC_avg_cost = sum([running_cost(x_true_rec[i][k],u_rec[i][k]) for i in 1:num_simulations, k in 1:T])/ (num_simulations * T)
    # linearMPC_avg_cost = sum([running_cost(x_true_rec_mpc[i][k],u_rec_mpc[i][k]) for i in 1:num_simulations, k in 1:T]) / (num_simulations * T)

    println("Average cost nonlinear MPC: ", sum(nonlinearMPC_cost)/num_simulations)
    println("Average cost linear MPC: ", sum(linearMPC_cost)/num_simulations)
    println("Average cost change: % ", (sum(nonlinearMPC_cost) - sum(linearMPC_cost)) / sum(linearMPC_cost) * 100)

end

function compare_estimation_err()

    est_err_rec = [norm(x_rec[i][k] - x_true_rec[i][k]) for i in 1:num_simulations, k in 1:T]
    est_err_rec_mpc = [norm(x_rec_mpc[i][k] - x_true_rec_mpc[i][k]) for i in 1:num_simulations, k in 1:T]

    println("Average Achieved Estimation Error Change: % ", (sum(est_err_rec) - sum(est_err_rec_mpc)) / sum(est_err_rec_mpc) * 100)

end


# histograms()
# plot_OCV_SOC()
compare_covariance_trace()
# compare_costs()
# compare_estimation_err()