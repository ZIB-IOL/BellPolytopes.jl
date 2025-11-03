function build_callback(
        p,
        v,
        o,
        shr2,
        verbose,
        epsilon,
        shortcut,
        nb_increment_interval,
        callback_interval,
        hyperplane_interval,
        bound_interval,
        save,
        file,
        save_interval,
    )
    @assert !(0 < shortcut < 1)
    if isnan(shr2)
        bound_interval = typemax(Int)
    end
    if verbose > 3
        println("Intervals")
        println("    Print: ", callback_interval)
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
        _print_headers()
    end
    function callback(state, active_set, args...)
        if mod(state.t, nb_increment_interval) == 0
            state.lmo.lmo.nb += 1
        end
        if verbose && mod(state.t, callback_interval) == 0
            _print_callback(state.t, state, active_set)
        end
        if verbose_hyperplane && mod(state.t, hyperplane_interval) == 0
            a = -state.gradient # v*p+(1-v)*o-active_set.x
            b = dot(a, state.v) # local max found by the LMO
            if verbose
                @printf("v_c ≤ %f\n", (b - dot(a, o)) / (dot(a, p) - dot(a, o)))
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
        if shortcut > 0 && state.dual_gap < state.primal / shortcut
            if verbose
                _print_callback("Shortcut", state, active_set)
            end
            return false
        end
        if state.primal ≤ epsilon || state.dual_gap ≤ epsilon
            if verbose
                _print_callback("Last", state, active_set)
            end
            return false
        end
        return true
    end
    return callback
end

function _print_headers()
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

function _print_callback(it, state, active_set)
    @printf(
            stdout,
            "%s    %.4e    %.4e    %.4e    %.4e   %s    %s\n",
            lpad(it, 12),
            state.primal,
            state.dual_gap,
            state.time,
            state.t / state.time,
            lpad(length(active_set), 7),
            lpad(state.lmo.lmo.cnt, 7)
           )
end
