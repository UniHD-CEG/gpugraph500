

#ifdef __cplusplus
extern "C"
{
#endif
#include <stddef.h>


void POMP2_Init_regions()
{
}

size_t POMP2_Get_num_regions()
{
    return 0;
}

const char* POMP2_Get_opari2_version()
{
    return "1.1.2";
}

/* 
 * The following functions define the POMP2 library interface version
 * the instrumented code conforms with. The library interface version
 * is modeled after
 * https://www.gnu.org/software/libtool/manual/libtool.html#Versioning
 */

int POMP2_Get_required_pomp2_library_version_current()
{
    return 2;
}

int POMP2_Get_required_pomp2_library_version_revision()
{
    return 0;
}

int POMP2_Get_required_pomp2_library_version_age()
{
    return 0;
}

#ifdef __cplusplus
}
#endif
