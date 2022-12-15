using Plots
using Distributions

using Gen

function normalize(dist::Vector{Float64})
    return dist/sum(dist)
end

function predict_new_data(model, trace, new_xs::Vector{Float64}, param_addrs)
    # Copy parameter values from the inferred trace (`trace`)
    # into a fresh set of constraints.
    constraints = Gen.choicemap()
    for addr in param_addrs
        constraints[addr] = trace[addr]
    end

    # Run the model with new x coordinates, and with parameters
    # fixed to be the inferred values.
    (new_trace, _) = Gen.generate(model, (new_xs,), constraints)

    # Pull out the y-values and return them.
    ys = [new_trace[(:x, i)] for i=1:length(new_xs)]
    return ys
end

function render_trace(trace; show_data=true)
    # Pull out xs from the trace
    xs, = get_args(trace)

    xmin = minimum(xs)
    xmax = maximum(xs)

    # Pull out the return value, useful for plotting
    func = get_retval(trace)

    fig = plot()

    if show_data
        xs = [trace[(:x, i)] for i=1:length(xs)]
        xs_model = evaluate_function(func, xs[1], 1., length(xs))
        println(func)
        println(xs)
        println(xs_model)
        # Plot the data set
        scatter!(1:length(xs), xs_model, c="black", label=nothing)
    end

    return fig
end

function round_all(xs::Vector{Float64}; n=2)
    map(x -> round(x; digits=n), xs)
end

"""Sample a categorical variable with given weights."""
function sample_categorical(probs::Vector{Float64})
    u = rand()
    cdf = cumsum(probs)
    for (i, c) in enumerate(cdf)
        if u < c return i end
    end
end

function grid(renderer::Function, traces)
    Plots.plot(map(renderer, traces)...)
end
