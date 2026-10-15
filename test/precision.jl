@testset "Stable residual objective and BigFloat symmetry" begin
    x=[1e8+1e-6,1e8-1e-6];y=fill(1e8,2)
    reference=sum(abs2,Rational{BigInt}.(x).-Rational{BigInt}.(y))/2
    @test BP._residual_primal(x,y)≈Float64(reference)
    @test BP._residual_primal(x,y)>0
    @test dot(y,y)/2+dot(x,x)/2-dot(x,y)==0
    sx=FrankWolfe.SubspaceVector(copy(x),x,[2,3])
    sy=FrankWolfe.SubspaceVector(copy(y),y,[2,3])
    @test BP._residual_primal(sx,sy)≈sum([2,3].*(x-y).^2)/2
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
