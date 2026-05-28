module BoseModule #モジュールの作成

function Bose(ω,T)
    cot_val = coth(ω/(2*T))
    return 0.5*(cot_val-1)
end

end