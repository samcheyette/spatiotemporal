include("tree.jl")
include("utils.jl")

using Plots




function evaluate_function(func::Node, x0::Float64, t0::Float64, n::Int; noise=0.1)

    x = x0
    A = Vector{Float64}()
    append!(A, normal(x, noise))
    for i in 2:n
        x = eval_node(func, x, t0+i-1)
        append!(A, normal(x,noise))
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
            #@trace(VarX(), :X)
            node = VarX()

        elseif node_type == VAR_T
            #@trace(VarT(), :T)
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



@gen function model(xs::Vector{Float64})
    n = length(xs)

    func::Node = @trace(pcfg_prior(), :tree)
    #noise = @trace(0.1, :noise)
    noise = 0.1
    A=Vector{Float64}()
    x_model = xs[1]
    ({(:x,1)} ~ normal(x_model,noise))
    for t in 2:n
        x_model = eval_node(func,x_model,t) 
        ({(:x,t)} ~ normal(x_model,noise))
    end
    return func
end


# traces = [Gen.simulate(model, (xs,)) for _=1:12]
# grid(render_trace, traces)

@gen function random_node_path_unbiased(node::Node)
    p_stop = isa(node, LeafNode) ? 1.0 : 1/size(node)
    t = @trace(bernoulli(p_stop), :stop)
    if t
        return :tree

    else

        # p_left = size(node.left) / (size(node) - 1)
        # (next_node, direction) = @trace(bernoulli(p_left), :left) ? (node.left, :left) : (node.right, :right)
        # rest_of_path = @trace(random_node_path_unbiased(next_node), :rest_of_path)

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
        rest_of_path = @trace(random_node_path_unbiased(next_node), :rest_of_path)

        if isa(rest_of_path, Pair)
            return :tree => direction => rest_of_path[2]
        else
            return :tree => direction
        end
    end
end


@gen function random_node_path_root(node::Node)
    return :tree
end

function get_value_at_path(tree, path)
    if !(isa(path, Pair))
        return tree
    else
        if isa(path[2],Pair)
            choice = path[2][1]
        else
            choice = path[2]
        end

        if isa(tree, UnaryOpNode)
            return get_value_at_path(tree.arg, path[2])
        elseif isa(tree, BinaryOpNode)
            if choice == :left
                return get_value_at_path(tree.left, path[2])
            else
                return get_value_at_path(tree.right, path[2])     
            end           
        else
            if choice == :condition
                return get_value_at_path(tree.condition, path[2])
            elseif choice == :left
                return get_value_at_path(tree.left, path[2])     
            else
                return get_value_at_path(tree.right, path[2])     
            end  
        end

        # if choice == :arg
        #     return get_value_at_path(tree.arg, path[2])
        # elseif choice == :left
        #     return get_value_at_path(tree.left, path[2])
        # else
        #     return get_value_at_path(tree.right, path[2])
        # end
    end
end

@gen function regen_random_subtree(prev_trace)
    path = @trace(random_node_path_unbiased(get_retval(prev_trace)), :path)
    #@trace(pcfg_prior("EXPR"), :new_subtree)

    change_node = get_value_at_path(get_retval(prev_trace), path)
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
    #println(get_submap(fwd_assmt, :path))
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
    D = Dict("iter"=> [],"posterior"=> [], "time"=>[])

    if visualize
        fig = plot(1:length(xs), xs, color="black", ylim=(xmin-diff,xmax+diff))
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
            xs_model = evaluate_function(get_retval(trace), xs[1], 1., length(xs)+5)

            println(iter)
            println(trace[:tree])
            println(round_all(xs))
            println(round_all(xs_model))
            println("")
            if (visualize) && (iter > 15000)
                gui(scatter!(fig,1:length(xs)+5, xs_model,
                 c="red",alpha=0.25, label=nothing))
            end

            #println(trace[:likelihood])
        end

        #(trace, _) = mh(trace, select(:noise))
    end
    return D
end

# Initialize trace and extract variables.

function initialize_trace(xs::Vector{Float64})
    constraints = choicemap()
    for (i, x) in enumerate(xs)
        constraints[(:x,i)] = x
    end
    #constraints[:ys] = ys
    (trace, _) = generate(model, (xs,), constraints)
    return trace
end




ts = [1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]
#xs = map(t -> t+t*sin(t*π/4), ts)
#xs = map(t -> 1+ sin(pi*t/4), ts)
#xs = map(t -> 1+t*sin(t/4),ts)
xs = map(t -> t * mod(t, 2),ts)
#xs = map(t -> t*2, ts)
#xs = map(t -> 1.,ts)


#xs = [25.,25.,25.,25.,25.,25.,25.]
trace = initialize_trace(xs)

trace_dict = run_mcmc(trace,xs, 100000; visualize=true)

#println(trace)

using CSV
using DataFrames

df = DataFrame(trace_dict)


CSV.write("output/gen_mcmc_time.csv", df )