include("tree.jl")
include("utils.jl")

using CSV
using DataFrames
using Plots

function evaluate_function(func::Node, t0::Integer, x0::Float64, n::Integer; noise=0.1)
    A = Vector{Float64}()
    x = x0
    append!(A, normal(x, noise))
    for t in 1:n
        x = eval_node(func, x, t0+t)
        append!(A, normal(x, noise))
    end
    return A
end

@gen function pcfg_prior(which_dist="EXPR", mean_value=0, sd_value=10)

    node_dist = node_dists[which_dist]
    node_type = @trace(categorical(node_dist), :type)

    if which_dist == "EXPR"
        if node_type == NUMBER
            param = @trace(normal(mean_value, sd_value), :param)
            node = Number(param)


        elseif node_type == VAR_X
            node = VarX()

        elseif node_type == VAR_T
            node = VarT()

        elseif node_type == PLUS
            left = @trace(pcfg_prior(), :left)
            right = @trace(pcfg_prior(), :right)
            node = Plus(left, right)

        elseif node_type == TIMES
            left = @trace(pcfg_prior(), :left)
            right = @trace(pcfg_prior(), :right)
            node = Times(left, right)

        elseif node_type == MINUS
            left = @trace(pcfg_prior(), :left)
            right = @trace(pcfg_prior(), :right)
            node = Minus(left, right)

        elseif node_type == DIVIDE
            left = @trace(pcfg_prior(), :left)
            right = @trace(pcfg_prior(), :right)
            node = Divide(left, right)

        elseif node_type == MOD
            left = @trace(pcfg_prior(), :left)
            right = @trace(pcfg_prior(), :right)
            node = Mod(left, right)

        elseif node_type == SIN
            arg = @trace(pcfg_prior(), :arg)
            node = Sin(arg)

        elseif node_type == COS
            arg = @trace(pcfg_prior(), :arg)
            node = Cos(arg)

        elseif node_type == IF_THEN
            cond = @trace(pcfg_prior("BOOL"), :condition)
            e1 = @trace(pcfg_prior(), :left)
            e2 = @trace(pcfg_prior(), :right)
            node = If_Then(cond, e1, e2)

        else
            error("Unknown node type: $node_type")
        end

    elseif which_dist == "BOOL"
        if node_type == EQUALS
            left = @trace(pcfg_prior(), :left)
            right = @trace(pcfg_prior(), :right)
            node = Equals(left, right)
        elseif node_type == GT
            left = @trace(pcfg_prior(), :left)
            right = @trace(pcfg_prior(), :right)
            node = Greater(left, right)
        # unknown node type
        else
            error("Unknown node type: $node_type")
        end
    # unknown node type
    else
        error("Unknown rule type: $which_dist")
    end

    return node
end

@gen function model(t0::Integer, x0::Float64, n::Integer)
    n = length(xs)
    func::Node = @trace(pcfg_prior(), :tree)
    noise = 0.1
    x = x0
    for t in 1:n
        x = eval_node(func, x, t0+t)
        ({(:x, t0+t)} ~ normal(x, noise))
    end
    return func
end

@gen function random_node_path_unbiased(node::Node)
    p_stop = isa(node, LeafNode) ? 1.0 : 1/size(node)
    t = @trace(bernoulli(p_stop), :stop)
    if t
        return (:tree, node)
    else
        if isa(node, UnaryOpNode)
            (next_node, direction) = (node.arg, :arg)

        elseif isa(node, BinaryOpNode)

            p_left = size(node.left) / (size(node) - 1)
            (next_node, direction) = @trace(bernoulli(p_left), :dir) ? (node.left, :left) : (node.right, :right)

        elseif isa(node, TrinaryOpNode)
            probs = [size(node.condition), size(node.left), size(node.right)] ./ (size(node)-1)
            choice = @trace(categorical(probs), :dir)
            if choice == 1
                (next_node, direction) = (node.condition, :condition)
            elseif choice == 2
                (next_node, direction) = (node.left, :left)
            elseif choice == 3
                (next_node, direction) = (node.right, :right)
            end
        else
            node_type = typeof(node)
            error("Unknown node type: $node_type")
        end

        (rest_of_path, final_node) = {:rest_of_path} ~ random_node_path_unbiased(next_node)

        if isa(rest_of_path, Pair)
            return (:tree => direction => rest_of_path[2], final_node)
        else
            return (:tree => direction, final_node)
        end
    end
end


@gen function random_node_path_root(node::Node)
    return (:tree, node)
end

@gen function regen_random_subtree(prev_trace)
    (path, change_node) = @trace(random_node_path_unbiased(get_retval(prev_trace)), :path)
    if (typeof(change_node) in BOOLS_LIST)
        @trace(pcfg_prior("BOOL"), :new_subtree)
    elseif isa(change_node, Number)
        @trace(pcfg_prior("EXPR", change_node.param, 1), :new_subtree)
    else
        @trace(pcfg_prior("EXPR"), :new_subtree)
    end
    return path
end

function subtree_involution(trace, fwd_assmt::ChoiceMap, path_to_subtree, proposal_args::Tuple)
    # Need to return a new trace, a bwd_assmt, and a weight.
    model_assmt = get_choices(trace)
    bwd_assmt = choicemap()
    set_submap!(bwd_assmt, :path, get_submap(fwd_assmt, :path))
    set_submap!(bwd_assmt, :new_subtree, get_submap(model_assmt, path_to_subtree))
    new_trace_update = choicemap()
    set_submap!(new_trace_update, path_to_subtree, get_submap(fwd_assmt, :new_subtree))
    (new_trace, weight, _, _) =
        update(trace, get_args(trace), (NoChange(),), new_trace_update)
    (new_trace, bwd_assmt, weight)
end

function run_mcmc(trace, xs, iters::Int; visualize=false)
    xmin=minimum(xs)
    xmax=maximum(xs)
    diff = xmax-xmin+1
    (t0, x0, n) = get_args(trace)
    ts_plot = collect(t0:t0+n)
    xs_plot = vcat(x0, xs)
    D = Dict("iter"=> [],"posterior"=> [], "time"=>[])

    if visualize
        fig = plot(ts_plot, xs_plot, color="black", ylim=(xmin-diff,xmax+diff))
        gui(fig)
    end

    for iter=1:iters
        dt = @elapsed begin
            (trace, _) = mh(trace, regen_random_subtree, (), subtree_involution)
        end

        append!(D["iter"], iter)
        append!(D["posterior"], get_score(trace))
        append!(D["time"], dt)

        if iter % 2500 == 0
            func = get_retval(trace)
            xs_model = evaluate_function(func, t0, x0, n+5)
            println(iter)
            println(trace[:tree])
            println(round_all(xs))
            println(round_all(xs_model))
            println("")
            if (visualize) && (iter > 15000)
                ts_plot = t0:t0+n+5
                gui(scatter!(fig, ts_plot, xs_model, c="red", alpha=0.25, label=nothing))
            end
        end
    end
    return D
end

# Initialize trace and extract variables.
function initialize_trace(t0::Integer, x0::Float64, xs::Vector{Float64})
    constraints = choicemap()
    n = length(xs)
    for t in 1:n
        constraints[(:x, t0+t)] = xs[t]
    end
    model_args = (t0, x0, n)
    (trace, _) = generate(model, model_args, constraints)
    return trace
end

ts = 0:10

# xs = map(t -> t+t*sin(t*π/4), ts)
# xs = map(t -> 1+ sin(pi*t/4), ts)
# xs = map(t -> 1+t*sin(t/4),ts)
# xs = map(t -> t * ((mod(t,3)==1)),ts)
# xs = map(t -> t * mod(t, 2),ts)
# xs = map(t -> t*2, ts)
# xs = map(t -> 1.,ts)
xs = map(t -> t * mod(t, 2), ts)

t0 = ts[1]
x0 = Float64(xs[1])
xs_obs = Vector{Float64}(xs[2:end])

trace = initialize_trace(t0, x0, xs_obs)
trace_dict = run_mcmc(trace, xs_obs, 1000000; visualize=true)

df = DataFrame(trace_dict)

CSV.write("output/gen_mcmc_time.csv", df )
