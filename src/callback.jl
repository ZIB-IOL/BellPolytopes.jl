function build_callback(
        p,
        v,
        o,
        shr2,
        verbose,
        epsilon,
        renorm_interval,
        nb_increment_interval,
        callback_interval,
        hyperplane_interval,
        bound_interval,
        save,
        file,
        save_interval,
    )
    if isnan(shr2)
        bound_interval = typemax(Int)
    end
    if verbose > 3
        println("Intervals")
        println("    Print: ", callback_interval)
        println("   Renorm: ", renorm_interval)
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
    verbose = verbose ≥ 3
    verbose_hyperplane = (verbose || save) && hyperplane_interval != typemax(Int)
    verbose_bound = verbose && bound_interval != typemax(Int)
    if verbose
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
    function callback(state, active_set, args...)
        if mod(state.t, renorm_interval) == 0
            FrankWolfe.active_set_renormalize!(active_set)
            FrankWolfe.compute_active_set_iterate!(active_set)
        end
        if mod(state.t, nb_increment_interval) == 0
            state.lmo.lmo.nb += 1
        end
        if verbose && mod(state.t, callback_interval) == 0
            @printf(
                stdout,
                "%s    %.4e    %.4e    %.4e    %.4e   %s    %s\n",
                lpad(state.t, 12),
                state.primal,
                state.dual_gap,
                state.time,
                state.t / state.time,
                lpad(length(active_set), 7),
                lpad(state.lmo.lmo.cnt, 7)
            )
        end
        if verbose_hyperplane && mod(state.t, hyperplane_interval) == 0
            a = -state.gradient # v*p+(1-v)*o-active_set.x
            b = FrankWolfe.fast_dot(a, state.v) # local max found by the LMO
            if verbose
                @printf("v_c ≤ %f\n", b / FrankWolfe.fast_dot(a, p))
            end
            if save
                serialize(file * "_hyperplane.dat", (a, b))
            end
        end
        if verbose_bound && mod(state.t, bound_interval) == 0
            ν = 1 / (1 + norm(v * p + (1 - v) * o - active_set.x, 2))
            @printf("v_c ≥ %f (%f)\n", shr2 * ν * v, shr2 * v)
        end
        if save && mod(state.t, save_interval) == 0
            serialize(file * "_tmp.dat", ActiveSetStorage(active_set))
        end
        # if state.dual_gap < state.primal / 2
            # return false
        # end
        return state.primal > epsilon
    end
    return callback
end
