@testset "Final residual reporting and BigFloat symmetry" begin
    for prob in (false,true)
        p=prob ? fill(big"0.25",2,2,2,2) : BigFloat[1 0;0 1]
        q=reshape(collect(1:length(p)),size(p))
        deflate,inflate=BP.build_deflate_inflate_q(BigFloat,q)
        res=bell_frank_wolfe(p;prob,sym=true,deflate,inflate,v0=big"0.3",
                            max_iteration=5,nb=2,mode_last=-1,epsilon=big"1e-24")
        @test eltype(res[1])==BigFloat
        @test eltype(res[5].weights)==BigFloat
        target=prob ? big"0.3"*p .+ big"0.7"/4 : big"0.3"*p
        @test res[3]≈sum(abs2,res[1]-target)/2 atol=big"1e-70"
        @test res[3]>=0
    end
end

@testset "Final primal after precision conversion" begin
    p = [1.0 1.0; 1.0 -1.0]
    for TL in (Float64, BigFloat)
        res = bell_frank_wolfe(p; v0 = 0.25, sym = false, TL, mode = 1,
            max_iteration = 5, epsilon = Base.rtoldefault(Float64))
        residual = res[1] - TL(0.25) * TL.(p)
        @test res[3] isa TL
        @test res[3] ≈ sum(abs2, residual) / 2
        @test res[3] ≥ 0
    end
end
