import Random
using Distributions

using Gen

include("utils.jl")

"""Node in a tree representing a covariance function"""
abstract type Node end
abstract type LeafNode <: Node end
abstract type BinaryOpNode <: Node end
abstract type UnaryOpNode <: Node end
#abstract type BoolOpNode <: Node end
abstract type TrinaryOpNode <: Node end

"""
    size(::Node)

Number of nodes in the subtree rooted at this node.
"""
Base.size(::LeafNode) = 1
Base.size(node::TrinaryOpNode) = node.size

Base.size(node::BinaryOpNode) = node.size
Base.size(node::UnaryOpNode) = node.size
#Base.size(node::BoolOpNode) = node.size



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

"""If-then node"""
struct If_Then <: TrinaryOpNode
    condition::Node
    left::Node
    right::Node
    size::Int
end
If_Then(condition, left, right) = If_Then(condition,left,right,size(condition)+
                        size(left)+size(right)+1)
function eval_node(node::If_Then, x, t)
    condition = eval_node(node.condition, x, t)
    if condition
        eval_node(node.left,x,t)
    else
        eval_node(node.right,x,t)
    end
end

"""Equals node"""
struct Equals <: BinaryOpNode
    left::Node
    right::Node
    size::Int
end
Equals(left, right) = Equals(left, right, size(left)+size(right)+1)
function eval_node(node::Equals, x, t)
    return eval_node(node.left,x,t) == eval_node(node.right,x,t)
end


"""Equals node"""
struct Greater <: BinaryOpNode
    left::Node
    right::Node
    size::Int
end
Greater(left, right) = Greater(left, right, size(left)+size(right)+1)
function eval_node(node::Greater, x, t)
    return eval_node(node.left,x,t) > eval_node(node.right,x,t)
end

################################################################

#EXPR
NUMBER = 1
VAR_X = 2
VAR_T = 3

PLUS = 4
MINUS = 5
TIMES = 6
DIVIDE = 7
MOD = 8
SIN = 9
COS = 10
IF_THEN = 11

#BOOL
EQUALS = 1
GT = 2

BOOLS_LIST = [Equals, Greater]

EXPRS = Dict(
    NUMBER  => 0,
    VAR_X   => 0,
    VAR_T   => 0,
    PLUS    => 2,
    MINUS   => 2,
    TIMES   => 2,
    DIVIDE  => 2,
    MOD     => 1,
    SIN     => 1,
     COS    => 1,
    IF_THEN => 3)

BOOLS = Dict(
    EQUALS  => 2,
    GT      => 2)

node_type_to_num_children = Dict(
            "EXPR" => EXPRS,
            "BOOL" => BOOLS)

node_dists = Dict{String, Vector{Float64}}();
#node_types = Dict{String, String}();
for key in keys(node_type_to_num_children)
    rules = node_type_to_num_children[key]
    node_dists[key] = Vector{Float64}()
    for rule_name in sort(collect(keys(rules)))
        n_children = rules[rule_name]
        append!(node_dists[key],4.0^-n_children)
    end

end

for key in keys(node_dists)
    node_dists[key] = normalize(node_dists[key])
end

println(node_dists)
