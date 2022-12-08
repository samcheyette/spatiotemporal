include("tree.jl")

using Distributions
using Statistics


############################################################
######### Inference utils ##################################
############################################################

MAX_BRANCH = 2

function evaluate_function(func::Node, x0::Float64, t0::Float64, n::Int)

    x = x0
    A = Vector{Float64}()
    append!(A, x)
    for i in 2:n
        x = eval_node(func, x, t0+i-1)
        append!(A, x)
    end
    return A
end

"""Return log likelihood given input/output values."""
function compute_log_likelihood(func::Node, noise::Float64,
       xs::Vector{Float64}, ts::Vector{Float64})
    lkhd = 0
    mu = eval_node(func, xs[1], ts[1])
    mu = xs[1]
    for i in 2:(length(ts))
        #println(func)
        mu = eval_node(func, mu, ts[i])

        lkhd = lkhd+ Distributions.logpdf(Distributions.Normal(mu, noise),xs[i])
    end
    return lkhd
end




"""Sample a categorical variable with given weights."""
function sample_categorical(probs::Vector{Float64})
    u = rand()
    cdf = cumsum(probs)
    for (i, c) in enumerate(cdf)
        if u < c return i end
    end
end


"""Return index of child node in a tree."""
function get_child(parent::Int, child_num::Int, max_branch::Int)
    @assert child_num >= 1 && child_num <= max_branch
    (parent - 1) * max_branch + child_num + 1
end




############################################################
######### Define tree ######################################
############################################################

struct Trace
    func::Node
    noise::Float64
    xs::Vector{Float64}
    ts::Vector{Float64}
    log_likelihood::Float64
end



function pcfg_prior()
    node_type = sample_categorical(node_dist)

    if node_type == NUMBER
        if rand() < 1/10
            param = π
        else
            param = rand(Distributions.DiscreteUniform(1,10))
        end
        node = Number(param)

    elseif node_type == VAR_T
        node = VarT()

    elseif node_type == VAR_X
        node = VarX()

    elseif node_type == PLUS
        left = pcfg_prior()
        right = pcfg_prior()
        node = Plus(left, right)

    elseif node_type == MINUS
        left = pcfg_prior()
        right = pcfg_prior()
        node = Minus(left, right)

    elseif node_type == TIMES
        left = pcfg_prior()
        right = pcfg_prior()
        node = Times(left, right)


    elseif node_type == DIVIDE
        left = pcfg_prior()
        right = pcfg_prior()
        node = Divide(left, right)


    elseif node_type == MOD
        left = pcfg_prior()
        right = pcfg_prior()
        node = Mod(left, right)

    elseif node_type == SIN
        arg = pcfg_prior()
        node = Sin(arg)

    elseif node_type == COS
        arg = pcfg_prior()
        node = Cos(arg)



    # unknown node type
    else
        error("Unknown node type: $node_type")
    end

    return node
end



############################################################
######### Inference ########################################
############################################################


"""
    pick_random_node_unbiased(::Node, cur::Int, max_branch::Int)

Return a random node in the subtree rooted at the given node, whose integer
index is given. The sampling is uniform at random over all nodes in the
tree.
"""

function replace_subtree(func::LeafNode, cur::Int, func2::Node, cur2::Int)
    return cur == cur2 ? func2 : func
end


function replace_subtree(func::UnaryOpNode, cur::Int, func2::Node, cur2::Int)
    if cur == cur2
        return func2
    end
    child = get_child(cur, 1, MAX_BRANCH)
    subtree = child == cur2 ? func2 :
        replace_subtree(func.arg, child, func2, cur2)
    return typeof(func)(subtree)
end

function replace_subtree(func::BinaryOpNode, cur::Int, func2::Node, cur2::Int)
    if cur == cur2
        return func2
    end
    child_l = get_child(cur, 1, MAX_BRANCH)
    child_r = get_child(cur, 2, MAX_BRANCH)
    subtree_left = child_l == cur2 ? func2 :
        replace_subtree(func.left, child_l, func2, cur2)
    subtree_right = child_r == cur2 ? func2 :
        replace_subtree(func.right, child_r, func2, cur2)
    return typeof(func)(subtree_left, subtree_right)
end

function pick_random_node_unbiased end


function pick_random_node_unbiased(node::LeafNode, cur::Int, max_branch::Int)
    return (cur, node)
end



# MH correction for unbiased sampling.
function get_alpha_subtree_unbiased(func_prev, func_prop)
    return log(size(func_prev)) - log(size(func_prop))
end


function pick_random_node_unbiased(node::UnaryOpNode, cur::Int, max_branch::Int)
    probs = [1, size(node.arg)] ./ size(node)
    choice = sample_categorical(probs)
    if choice == 1
        return (cur, node)
    elseif choice == 2
        n_child = get_child(cur, 1, max_branch)
        return pick_random_node_unbiased(node.arg, n_child, max_branch)
    else
        @assert false "Unexpected child node $(choice)"
    end
end



function pick_random_node_unbiased(node::BinaryOpNode, cur::Int, max_branch::Int)
    probs = [1, size(node.left), size(node.right)] ./ size(node)
    choice = sample_categorical(probs)
    if choice == 1
        return (cur, node)
    elseif choice == 2
        n_child = get_child(cur, 1, max_branch)
        return pick_random_node_unbiased(node.left, n_child, max_branch)
    elseif choice == 3
        n_child = get_child(cur, 2, max_branch)
        return pick_random_node_unbiased(node.right, n_child, max_branch)
    else
        @assert false "Unexpected child node $(choice)"
    end
end



function mh_resample_subtree_unbiased(prev_trace)
    (loc_delta, node_delta) = pick_random_node_unbiased(prev_trace.func, 1, MAX_BRANCH)
    subtree = pcfg_prior()
    func_new = replace_subtree(prev_trace.func, 1, subtree, loc_delta)
    log_likelihood = compute_log_likelihood(func_new, prev_trace.noise,
        prev_trace.xs, prev_trace.ts)
    new_trace = Trace(func_new, prev_trace.noise, prev_trace.xs,
        prev_trace.ts, log_likelihood)
    alpha_size = get_alpha_subtree_unbiased(node_delta, subtree)
    alpha_ll = new_trace.log_likelihood - prev_trace.log_likelihood
    alpha = alpha_ll + alpha_size
    return log(rand()) < alpha ? new_trace : prev_trace
end

function mh_resample_subtree_root(prev_trace)
    func_new = pcfg_prior()
    log_likelihood = compute_log_likelihood(func_new, prev_trace.noise,
        prev_trace.xs, prev_trace.ts)
    new_trace = Trace(func_new, prev_trace.noise, prev_trace.xs,
        prev_trace.ts, log_likelihood)
    alpha = new_trace.log_likelihood - prev_trace.log_likelihood
    return log(rand()) < alpha ? new_trace : prev_trace
end



function initialize_trace(xs::Vector{Float64}, ts::Vector{Float64})
    func::Node = pcfg_prior()
    noise = 0.1
    log_likelihood = compute_log_likelihood(func, noise, xs, ts)
    return Trace(func, noise, xs, ts, log_likelihood)
end


function run_mcmc(prev_trace, iters::Int)
    new_trace = prev_trace

    x0 = prev_trace.xs[1]
    t0 = prev_trace.ts[1]
    for iter=1:iters
        new_trace = mh_resample_subtree_unbiased(new_trace)
       # new_trace = mh_resample_noise(new_trace)
        if iter % 10000 == 0
            println(iter)
            println(new_trace.func, " ", new_trace.log_likelihood)
            println(prev_trace.xs)
            println(evaluate_function(new_trace.func, x0,t0, 10))
            println("")
        end
    end
    return new_trace
end


#xs = [1.,4.,9.,16.,25.,36.]
#xs = [2.,4.,6.,8.,10.,12.]
#xs = [-1.,0.,1.,2.,3.,4.]
#xs = [1.,2.,3.,4.,5.,6.]
ts = [1.,2.,3.,4.,5.,6.,7.,8.]
xs = map(t -> t*sin(t/2), ts)
#xs = [25.,25.,25.,25.,25.,25.,25.]
trace = initialize_trace(xs, ts)
run_mcmc(trace, 10000000)