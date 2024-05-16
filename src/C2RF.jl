module C2RF

using Gurobi,JuMP,LinearAlgebra,SCIP,GroupSlices,Statistics,StatsBase
export C2RFf
export pC2RFmain
export pC2RFA
export pC2RFB



function C2RFf(R, λ, ℓ, u, maxtime,solver=1)
    #R = Matrix of classifications of the decision trees
    #λ = cardinality constraint
    #ℓ, u = bounds on α
    #solver = 1: Gurobi, else SCIP
    if solver == 1
        model = Model(optimizer_with_attributes(Gurobi.Optimizer))
        set_optimizer_attribute(model, "MIPFocus", 3)
    else 
        model = Model(optimizer_with_attributes(SCIP.Optimizer))
        gap = NaN
    end 
    m, c = size(R)
    ηbar = max(m-λ,λ)
    M = u * c + 1
    set_time_limit_sec(model, maxtime)
       @variable(model, 0 ≤ η ≤ ηbar)
    @variable(model, ℓ ≤ α[j=1:c] ≤ u)
    @variable(model, z[1:m], Bin)
    @constraint(model, sum(z) ≤ λ + η)
    @constraint(model, sum(z) ≥ λ - η)
    @constraint(model, [i = 1:m], dot(α, R[i, :]) ≥ 1 - (1 - z[i]) * M)
    @constraint(model, [i = 1:m], dot(α, R[i, :]) ≤ -1 + z[i] * M)
    @objective(model, Min, η)
    optimize!(model)
    if solver == 1
        gap = MOI.get(model, MOI.RelativeGap())
    end
    α, fun = value.(α), objective_value(model)
    classf2 = value.(z)
    return classf2, α, fun, gap
end

function pC2RFA(R, ℓ, u, λ, maxtime)
    #step 1: R2 = \bar{R} as line 1
    m, t = size(R)
    R2 = unique(R, dims=1)
    Ru = groupslices(R, dims=1)
    Ru2 = unique(Ru)
    lRu = length(Ru2)
    #step 2: compute w as in line 3
    w = zeros(lRu)
    for i = 1:lRu
        w[i] = length(findall(x -> x == Ru2[i], Ru))
    end
    #step 2: compute φ and ϕ as in line 3
    removbin = Int64.(zeros(0, 2))
    ### removbin[i,:] = [index of point and  binary variable fixed]
    notremove = []
    for i = 1:size(R2)[1]
        try
            global bi = countmap(R2[i, :])[-1]
        catch
            global bi= 0
        end
        try
            global ci = countmap(R2[i, :])[1]
        catch
            global ci = 0
        end
        if -u * bi+ ℓ * ci ≥ 1
            removbin = [removbin; i 1]
        elseif u * ci - ℓ * bi≤ -1
            removbin = [removbin; i 0]
        else
            push!(notremove, i)
        end

    end
    # new  \bar{\lambda} and  \bar{\eta}
    # λ0 = \bar{\lambda}
    # η0 = \bar{\eta}
    m0 = m - sum(w[removbin[:, 1]])
    imp = dot(w[removbin[:, 1]], removbin[:, 2])
    λ0 = λ - imp
    if λ0 < 0
        println("λ <0")
        λ0 = 0
    end
    η0 = max(m0 - λ0, λ0)
    ### only points consider in the optimization problem: i ∈ [1,h]\B as in (P3b) and (P3c)
    R2 = R2[notremove, :]
    w = w[notremove]

    ###step 10
    Ru2 = groupslices(R2, dims=2)
    lRu2 = unique(Ru2)
    sR2 = length(lRu2)
    ℓbound = zeros(sR2)
    indα = Int64.(zeros(t))
    for i = 1:sR2
        wf = findall(x -> x == lRu2[i], Ru2)
        lwf = length(wf)
        indα[wf] = i * ones(lwf)
        ℓbound[i] = ℓ * lwf
    end
    R2 = unique(R2, dims=2) ##finalmatrix
    for i = 1:sR2
        R2[:, i] = ℓbound[i] * R2[:, i]
    end

  
    ####ordering: step 15
    value2 = abs.((mean(R2, dims=2)))[:, 1]
    order = sortperm(value2,rev=false)
    order2 = zeros(length(order))
    for i = 1 : length(order)
       order2[i] = findfirst(x-> x==i, order)
       end
    order= Int64.(order2)
    return R2, w,  η0, λ0, order, indα



end




function pC2RFmain(R, λ, ℓ, u, maxtime,solver=1)
    #R = Matrix of classifications of the decision trees
    #λ = cardinality constraint
    #ℓ, u = bounds on α
    #solver = 1: Gurobi, else SCIP
    m,t = size(R)
    M = u*t+1
    ti = @elapsed R2, w, η0, λ0, order, indα = pC2RFA(R, ℓ, u, λ, maxtime)
    class, α, fun, gap = pC2RFB(R2, w, η0, λ0, ℓ, u, order, M,maxtime - ti,solver)
    m, t = size(R)
    fα = zeros(t)
    for i = 1:t
        fα[i] = α[indα[i]]
    end
    classF = zeros(m)
    for i = 1:m
        if dot(fα, R[i, :]) > 0
            classF[i] = 1
        end
    end
    FUN = norm(λ - sum(classF))
    return classF, fα, FUN, gap

end
#########OBSERVE THAT SCIP DOES NOT CONSIDER THE BRANCHING PRIORITY AS GUROBI
function pC2RFB(R2, w, η0, λ, ℓ, u, order,M, maxtime,solver)
    if solver == 1
        model = Model(optimizer_with_attributes(Gurobi.Optimizer))
        set_optimizer_attribute(model, "MIPFocus", 3)
    else 
        model = Model(optimizer_with_attributes(SCIP.Optimizer))
        gap = NaN
    end 
    m, t = size(R2)
    set_time_limit_sec(model, maxtime)
    @variable(model, 0 ≤ η ≤ η0)
    @variable(model, ℓ ≤ α[j=1:t] ≤ u)
    @variable(model, z[1:m], Bin)
    if solver == 1 
        for i = 1:m
            MOI.set(model, Gurobi.VariableAttribute("BranchPriority"), z[i], order[i])
        end
    else
         
    end
    @constraint(model, dot(z, w) ≤ λ + η)
    @constraint(model, dot(z, w) ≥ λ - η)
    @constraint(model, [i = 1:m], dot(α, R2[i, :]) ≥ 1 - (1 - z[i]) * M)
    @constraint(model, [i = 1:m], dot(α, R2[i, :]) ≤ -1 + z[i] * M)
    @objective(model, Min, η)
    optimize!(model)
    α, fun = value.(α), objective_value(model)
    classf = value.(z)
    if solver == 1
        gap = MOI.get(model, MOI.RelativeGap())
    end
    return classf, α, fun, gap

end


end # module
