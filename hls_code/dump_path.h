#ifndef DUMP_PATH_H
#define DUMP_PATH_H

// Global dump output path — set by testbench before calling RoFormer_Top()
// Guards ensure this only exists during C-simulation, not synthesis
#ifndef __SYNTHESIS__
#include <string>

// Declared here, defined in the testbench .cpp
extern std::string g_dump_prefix;

#endif // __SYNTHESIS__
#endif // DUMP_PATH_H
