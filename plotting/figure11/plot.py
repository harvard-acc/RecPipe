from mpl_plot import *

def area():
    print("Figure 11 Area synthesis results summary (12nm):")

    #units: um^2
    bs_sa = 506891   # baseline 32X32 systolic array, from MMU-compile.area
    ms_sa = 527807   # PRAccel 32X32 systolic array with 4X4 minimal sub-arrays, from DYN_MMU-compile.area
    simd = 106       # accumulators, from simd-compile.area
    crossbar = 287   # 4x4 crossbar, from crossbar4x4-compile.area
    filtering = 453  # filting unit, from sortvals-compile.area
    SRAM_32K = 29142 # 32K high-speed single port, 16384 words X 16 bits, mux width 16, bank 4
    SRAM_4K = 4684   # 4K high-density single port, 2048 words X 16 bits, mux width 4, bank 1
    SRAM_256K = 183290 #256K High-density single port, 65536 words, 32 output bits, mux width 32, bank 8


    print("baseline 32X32 systolic array", bs_sa, "mm2")
    print("PRAccel 32X32 systolic array", ms_sa, "mm2")
    print("simd accumulators", simd, "mm2")
    print("4X4 crossbar", crossbar, "mm2")
    print("filtering", filtering, "mm2")
    print("4K SRAM Marco", SRAM_4K, "mm2")
    print("32K SRAM Marco", SRAM_32K, "mm2")
    print("256K SRAM Marco", SRAM_256K, "mm2")

    # Scaling up to a 128X128 Systolic array with 4M activation, 4M weight, and 16M Embedding SRAM
    #           = [128X128 SA + simd      + crossbar      , Filtering      , 4M Activation , 4M Weight   , 16M Embedding]
    baseline    = [ bs_sa*16 + simd*128   + 0             , 0              , SRAM_32K*128  , SRAM_256K*16, SRAM_256K*64]
    multi_stage = [ ms_sa*16 + simd*128*8 + crossbar*16*16, filtering*32*32, SRAM_4K*1024  , SRAM_256K*16, SRAM_256K*64]

    ms_sa = multi_stage[0]
    bs_sa = baseline[0]

    print(ms_sa/ bs_sa)

    baseline_area = sum(baseline)
    multi_stage_area = sum(multi_stage)
    print("Increase in area", multi_stage_area / baseline_area)
    print("Multi-stage area", multi_stage_area / 1000 / 1000, "mm2")
    size = 0.35

    for i in range(len(baseline)):
        multi_stage[i] = multi_stage[i] - baseline[i]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    filename = "area"

    total = sum(baseline) + sum(multi_stage)
    data = baseline + multi_stage
    total = sum(data)
    data = list(map(lambda x: int(100*(x/float(total))), data))

    colors = [crimsons[0], blues[1], greens[1], oranges[1], purples[1], crimsons[1], blues[0], greens[0], oranges[0], purples[0]]
    wedge = (0., 0., 0.,0.,0., 0.5, 0.5, 0.5, 0.5, 0.5)

    outerpie  = [ sum(data[:5]), sum(data[5:]) ]

    ax1.pie(outerpie, colors=['lightgrey', black], radius=1-size,
            wedgeprops=dict(width=size, edgecolor='w', linewidth=0.1))

    ax1.pie(data, colors=colors, radius=1,
            wedgeprops=dict(width=size, edgecolor='w', linewidth=0.1))

    plt.savefig(filename+'.pdf')


def power():
    print("Figure 11 Power synthesis results summary (12nm):")
    # units: mW
    bs_sa = 166   # baseline 32X32 systolic array, from MMU-compile.power
    ms_sa = 171   # PRAccel 32X32 systolic array, from DYN_MMU-compile.power
    simd = 0.14       # accumulators, from simd-compile.power
    crossbar = 0.32   # 4x4 crossbar, from crossbar4x4-compile.power
    filtering = 0.198  # filting unit, from sortvals-compile.power
    SRAM_32K = 2.4 # 32K high-speed single port, 16384 words X 16 bits, mux width 16, bank 4
    SRAM_4K = 0.8   # 4K high-density single port, 2048 words X 16 bits, mux width 4, bank 1
    SRAM_256K = 8.4 #256K High-density single port, 65536 words, 32 output bits, mux width 32, bank 8

    print("baseline 32X32 systolic array", bs_sa, "mW")
    print("PRAccel 32X32 systolic array", ms_sa, "mW")
    print("simd accumulators", simd, "mW")
    print("4X4 crossbar", crossbar, "mW")
    print("filtering", filtering, "mW")
    print("4K SRAM Marco", SRAM_4K, "mW")
    print("32K SRAM Marco", SRAM_32K, "mW")
    print("256K SRAM Marco", SRAM_256K, "mW")

    # Scaling up to a 128X128 Systolic array with 4M activation, 4M weight, and 16M Embedding SRAM
    #           = [128X128 SA + simd      + crossbar      , Filtering      , 4M Activation , 4M Weight   , 16M Embedding]
    baseline    = [ bs_sa*16 + simd*128   + 0             , 0              , SRAM_32K*128  , SRAM_256K*16, SRAM_256K*64]
    multi_stage = [ ms_sa*16 + simd*128*8 + crossbar*16*16, filtering*32*32, SRAM_4K*1024  , SRAM_256K*16, SRAM_256K*64]


    ms_sa = multi_stage[0]
    bs_sa = baseline[0]


    baseline_power = sum(baseline)
    multi_stage_power = sum(multi_stage)

    print("Systolic array increase", ((ms_sa-bs_sa)/ baseline_power * 100.))

    print("Filtering "        , ((multi_stage[1] - baseline[1]) / baseline_power * 100.))
    print("Activation "       , ((multi_stage[2] - baseline[2]) / baseline_power * 100.))
    print("Weights "          , ((multi_stage[3] - baseline[3]) / baseline_power * 100.))
    print("Emb "              , ((multi_stage[4] - baseline[4]) / baseline_power * 100.))
    print("Increase in power" , multi_stage_power / baseline_power)
    print("Multi-stage power" , multi_stage_power                                , "mW")
    size = 0.35

    for i in range(len(baseline)):
        multi_stage[i] = multi_stage[i] - baseline[i]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    filename = "power"

    total = sum(baseline) + sum(multi_stage)
    data = baseline + multi_stage
    total = sum(data)
    data = list(map(lambda x: int(100*(x/float(total))), data))

    colors = [crimsons[0], blues[1], greens[1], oranges[1], purples[1], crimsons[1], blues[0], greens[0], oranges[0], purples[0]]
    wedge = (0., 0., 0., 0., 0., 0.5, 0.5, 0.5, 0.5, 0.5)

    outerpie  = [ sum(data[:5]), sum(data[5:]) ]

    ax1.pie(outerpie, colors=['lightgrey', black], radius=1-size,
            wedgeprops=dict(width=size, edgecolor='w', linewidth=0.1))

    ax1.pie(data, colors=colors, radius=1,
            wedgeprops=dict(width=size, edgecolor='w', linewidth=0.1))

    plt.savefig(filename+'.pdf')
if __name__=="__main__":
    area()
    power()
