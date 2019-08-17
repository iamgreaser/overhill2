OverHill2: An obscenely fast Silent Hill 2 RNG seed grinder
Copyright (c) GreaseMonkey, 2019
See the LICENCE.txt file this came with.
(Alternatively, imagine the zlib licence with my name on it.)

Special thanks to sh2_luck for their research into how the SH2 RNG
works, for showing it off in the first place, for providing a table
that the community could make good use of, and for helping me get
my head around what the randomisation was actually doing internally.

Given a starting clock time and a carbon code, this can produce a
full set of seeds in under half a second on my i5-6500.

You will need SSE4.1 to compile this.
For extra performance, enable AVX purely so you can get VEX opcodes.
(The 256-bit stuff AVX adds is actually slower by this point.)


