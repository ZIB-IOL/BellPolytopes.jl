function build_callback(
    trajectory_arr,
    p,
    v,
    o,
    shr2,
    verbose,
    epsilon,
    callback_interval,
    renorm_interval,
    reduce_interval,
    hyperplane_interval,
    bound_interval,
    nb_increment_interval,
    save,
    file,
    save_interval,
)
    if isnan(shr2)
        bound_interval = typemax(Int)
    end
    if verbose ≥ 3
        println("Intervals")
        println("    Print: ", callback_interval)
        println("   Renorm: ", renorm_interval)
        println("   Reduce: ", reduce_interval)
        if hyperplane_interval != typemax(Int)
            println("    Upper: ", hyperplane_interval)
        end
        if bound_interval != typemax(Int)
            println("    Lower: ", bound_interval)
        end
        if nb_increment_interval != typemax(Int)
            println("Increment: ", nb_increment_interval)
        end
        if save
            println("     Save: ", save_interval)
        end
        println()
    end
    return function callback(state, args...)
        if length(args) > 0
            if verbose ≥ 3 && state.t == 1
                @printf(
                    stdout,
                    "%s    %s    %s    %s    %s   %s    %s\n",
                    lpad("Iteration", 12),
                    lpad("Primal", 10),
                    lpad("Dual gap", 10),
                    lpad("Time (sec)", 10),
                    lpad("#It/sec", 10),
                    lpad("#Atoms", 7),
                    lpad("#LMO", 7)
                )
            end
            active_set = args[1]
            push!(trajectory_arr, (FrankWolfe.callback_state(state)..., length(active_set)))
            state.v.lmo.data[1] += 1
            if verbose ≥ 3 && mod(state.t, callback_interval) == 0
                @printf(
                    stdout,
                    "%s    %.4e    %.4e    %.4e    %.4e   %s    %s\n",
                    lpad(state.v.lmo.data[1], 12),
                    state.primal,
                    state.dual_gap,
                    state.time,
                    state.t / state.time,
                    lpad(length(active_set), 7),
                    lpad(state.v.lmo.data[2], 7)
                )
            end
            if mod(state.t, renorm_interval) == 0
                FrankWolfe.active_set_renormalize!(active_set)
                FrankWolfe.compute_active_set_iterate!(active_set)
            end
            if mod(state.t, reduce_interval) == 0
                active_set_reduce_dot!(active_set, state.v)
            end
            if mod(state.t, hyperplane_interval) == 0
                a = -state.gradient # v*p+(1-v)*o-active_set.x
                b = FrankWolfe.fast_dot(a, state.v) # local max found by the LMO
                if verbose ≥ 3
                    @printf("v_c ≤ %f\n", b / FrankWolfe.fast_dot(a, p))
                end
                if save
                    serialize(file * "_hyperplane.dat", (a, b))
                end
            end
            if mod(state.t, bound_interval) == 0
                ν = 1 / (1 + norm(v * p + (1 - v) * o - active_set.x, 2))
                if verbose ≥ 3
                    @printf("v_c ≥ %f (%f)\n", shr2 * ν * v, shr2 * v)
                end
            end
            if mod(state.t, nb_increment_interval) == 0
                state.v.lmo.nb += 1
                #  println("The heuristic now runs ", state.v.lmo.nb, " times")
            end
            if save && mod(state.t, save_interval) == 0
                #  serialize(file * "_tmp.dat", ActiveSetStorage(active_set))
                serialize(file * "_tmp.dat", active_set)
            end
        end
        return state.primal > epsilon
    end
end
