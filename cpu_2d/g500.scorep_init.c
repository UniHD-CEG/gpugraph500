#include <stddef.h>

extern const struct SCOREP_Subsystem SCOREP_Subsystem_MetricService;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_CompilerAdapter;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_MpiAdapter;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_PompOmpAdapter;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_ThreadForkJoin;

const struct SCOREP_Subsystem* scorep_subsystems[] = {
    &SCOREP_Subsystem_MetricService,
    &SCOREP_Subsystem_CompilerAdapter,
    &SCOREP_Subsystem_MpiAdapter,
    &SCOREP_Subsystem_PompOmpAdapter,
    &SCOREP_Subsystem_ThreadForkJoin
};

const size_t scorep_number_of_subsystems = sizeof( scorep_subsystems ) /
                                           sizeof( scorep_subsystems[ 0 ] );
