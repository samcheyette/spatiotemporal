import LinearAlgebra
import Random
using Gen

"""Node in a tree representing a covariance function"""
abstract type Node end
abstract type LeafNode <: Node end
abstract type BinaryOpNode <: Node end
abstract type UnaryOpNode <: Node end


"""
    size(::Node)

Number of nodes in the subtree rooted at this node.
"""
Base.size(::LeafNode) = 1
Base.size(node::BinaryOpNode) = node.size
Base.size(node::UnaryOpNode) = node.size



"""Variables and numbers"""
struct Number <: LeafNode
    param::Float64
end

eval_node(node::Number, x, t) = node.param


"""Variables and numbers"""
struct VarT <: LeafNode end
eval_node(node::VarT, x, t) = t;


"""Variables and numbers"""
struct VarX <: LeafNode end
eval_node(node::VarX, x, t) = x;


"""Plus node"""
struct Plus <: BinaryOpNode
    left::Node
    right::Node
    size::Int
end
Plus(left, right) = Plus(left, right, size(left) + size(right) + 1)
function eval_node(node::Plus, x, t)
    eval_node(node.left, x, t) + eval_node(node.right, x, t)
end


"""Minus node"""
struct Minus <: BinaryOpNode
    left::Node
    right::Node
    size::Int
end
Minus(left, right) = Minus(left, right, size(left) + size(right) + 1)
function eval_node(node::Minus, x, t)
    eval_node(node.left, x, t) - eval_node(node.right, x, t)
end


"""Times node"""
struct Times <: BinaryOpNode
    left::Node
    right::Node
    size::Int
end
Times(left, right) = Times(left, right, size(left) + size(right) + 1)
function eval_node(node::Times, x, t)
    eval_node(node.left, x, t) * eval_node(node.right, x, t)
end


"""Divide node"""
struct Divide <: BinaryOpNode
    left::Node
    right::Node
    size::Int
end
Divide(left, right) = Divide(left, right, size(left) + size(right) + 1)
function eval_node(node::Divide, x, t)
    divisor = eval_node(node.right, x, t)
    if divisor == 0
        return Inf
    else
        return  eval_node(node.left, x, t) / divisor
    end
end

"""Mod node"""
struct Mod <: BinaryOpNode
    left::Node
    right::Node
    size::Int
end
Mod(left, right) = Mod(left, right, size(left) + size(right) + 1)
function eval_node(node::Mod, x, t)
    divisor = eval_node(node.right, x, t)
    if divisor == 0
        return Inf
    else
        return  eval_node(node.left, x, t) % divisor
    end
end


"""Sine node"""
struct Sin <: UnaryOpNode
    arg::Node
    size::Int
end
Sin(arg) = Sin(arg, size(arg)+1)
function eval_node(node::Sin, x, t)
    arg = eval_node(node.arg, x, t)
    if abs(arg) == Inf
        return Inf
    else 
        return sin(arg)
    end
end

"""Cosine node"""
struct Cos <: UnaryOpNode
    arg::Node
    size::Int
end
Cos(arg) = Cos(arg, size(arg)+1)
function eval_node(node::Cos, x, t)
    arg = eval_node(node.arg, x, t)
    if abs(arg) == Inf
        return Inf
    else 
        return cos(arg)
    end
end




################################################################

function normalize(dist::Vector{Float64})
    return dist/sum(dist)
end



const NUMBER = 1
const VAR_X = 2
const VAR_T = 3
const PLUS = 4
const MINUS = 5
const TIMES = 6
const DIVIDE = 7
const MOD = 8
const SIN = 9
const COS = 10

node_type_to_num_children = Dict(
    NUMBER => 0,
    VAR_X => 0,
    VAR_T => 0,
    PLUS => 2,
    MINUS => 2,
    TIMES => 2,
    DIVIDE => 2,
    MOD => 2, 
    SIN => 1,
    COS => 1)


node_dist = Vector{Float64}()
for key in sort(collect(keys(node_type_to_num_children)))
    n_children = node_type_to_num_children[key]
    append!(node_dist, 2.0^-n_children)
end

node_dist = normalize(node_dist)

const MAX_BRANCH = 2

#################################################################

