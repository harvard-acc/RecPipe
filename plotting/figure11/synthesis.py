# This script generates Area and Power breakdown of RPAccel syntheszied with 12nm technology node.
# The synthesis reports are in /RecPipe-Artifact/RPAccel_characterization/Synthesis_Report/

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

    baseline_area = sum(baseline)
    multi_stage_area = sum(multi_stage)
    
    print("Total Area with 128X128 systolic array, 4M activation, 4M weight, and 16M Embedding SRAM:")

    print("Total Baseline TPU-like accelerator area", baseline_area / 1000 / 1000, "mm2")
    print("Total PRAccel accelerator area", multi_stage_area / 1000 / 1000, "mm2")
    print("Area overhead (128X128 SA)", (multi_stage_area-baseline_area) / float(baseline_area))
    print('\n')

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

    

    baseline_power = sum(baseline)
    multi_stage_power = sum(multi_stage)
    
    print("Total Power with 128X128 systolic array, 4M activation, 4M weight, and 16M Embedding SRAM:")
    print("Total Baseline TPU-like accelerator area", baseline_power, "mW")
    print("Total PRAccel accelerator power", multi_stage_power, "mW")
    print("Power overhead (128X128 SA)", (multi_stage_power-baseline_power) / float(baseline_power))
    print('\n')
   
if __name__=="__main__":


    area()
    power()
