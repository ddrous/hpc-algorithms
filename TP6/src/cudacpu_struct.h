#ifndef CUDACPU_STRUCT_H
#define CUDACPU_STRUCT_H

#include <cstddef>

//////////////////////////////////////////////////////////////////////////////////////

template <class IdxType>
struct cudacpu_1D_struct{
    IdxType x;
};

template <class IdxType>
cudacpu_1D_struct<IdxType> cudacpu_make_1D(const IdxType& inX){
    return  cudacpu_1D_struct<IdxType>{inX};
}

template <class IdxType>
struct cudacpu_2D_struct{
    IdxType x;
    IdxType y;
};

template <class IdxType>
cudacpu_2D_struct<IdxType> cudacpu_make_2D(const IdxType& inX, const IdxType& inY){
    return  cudacpu_2D_struct<IdxType>{inX, inY};
}

template <class IdxType>
struct cudacpu_3D_struct{
    IdxType x;
    IdxType y;
    IdxType z;
};

template <class IdxType>
cudacpu_3D_struct<IdxType> cudacpu_make_3D(const IdxType& inX, const IdxType& inY, const IdxType& inZ){
    return  cudacpu_3D_struct<IdxType>{inX, inY, inZ};
}

template <class IdxType>
struct cudacpu_4D_struct{
    IdxType x;
    IdxType y;
    IdxType z;
    IdxType w;
};

template <class IdxType>
cudacpu_4D_struct<IdxType> cudacpu_make_4D(const IdxType& inX, const IdxType& inY, const IdxType& inZ, const IdxType& inW){
    return  cudacpu_4D_struct<IdxType>{inX, inY, inZ, inW};
}

//////////////////////////////////////////////////////////////////////////////////////

using cudaPos = cudacpu_3D_struct<size_t>;
#define make_cudaPos cudacpu_make_3D<size_t>

// for type in char uchar short ushort int uint long ulong float longlong ulonglong double ; do
//    for size in 1 2 3 4 ; do
//        echo "using $type$size = cudacpu_"$size"D_struct<$type>;"
//        echo "#define make_$type$size cudacpu_make_"$size"D<$type>"
//    done
// done
// char1, uchar1, short1, ushort1, int1, uint1, long1, ulong1, float1
// char2, uchar2, short2, ushort2, int2, uint2, long2, ulong2, float2
// char3, uchar3, short3, ushort3, int3, uint3, long3, ulong3, float3
// char4, uchar4, short4, ushort4, int4, uint4, long4, ulong4, float4

// longlong1, ulonglong1, double1
// longlong2, ulonglong2, double2

using char1 = cudacpu_1D_struct<char>;
#define make_char1 cudacpu_make_1D<char>
using char2 = cudacpu_2D_struct<char>;
#define make_char2 cudacpu_make_2D<char>
using char3 = cudacpu_3D_struct<char>;
#define make_char3 cudacpu_make_3D<char>
using char4 = cudacpu_4D_struct<char>;
#define make_char4 cudacpu_make_4D<char>
using uchar1 = cudacpu_1D_struct<unsigned char>;
#define make_uchar1 cudacpu_make_1D<unsigned char>
using uchar2 = cudacpu_2D_struct<unsigned char>;
#define make_uchar2 cudacpu_make_2D<unsigned char>
using uchar3 = cudacpu_3D_struct<unsigned char>;
#define make_uchar3 cudacpu_make_3D<unsigned char>
using uchar4 = cudacpu_4D_struct<unsigned char>;
#define make_uchar4 cudacpu_make_4D<unsigned char>
using short1 = cudacpu_1D_struct<short>;
#define make_short1 cudacpu_make_1D<short>
using short2 = cudacpu_2D_struct<short>;
#define make_short2 cudacpu_make_2D<short>
using short3 = cudacpu_3D_struct<short>;
#define make_short3 cudacpu_make_3D<short>
using short4 = cudacpu_4D_struct<short>;
#define make_short4 cudacpu_make_4D<short>
using ushort1 = cudacpu_1D_struct<unsigned short>;
#define make_ushort1 cudacpu_make_1D<unsigned short>
using ushort2 = cudacpu_2D_struct<unsigned short>;
#define make_ushort2 cudacpu_make_2D<unsigned short>
using ushort3 = cudacpu_3D_struct<unsigned short>;
#define make_ushort3 cudacpu_make_3D<unsigned short>
using ushort4 = cudacpu_4D_struct<unsigned short>;
#define make_ushort4 cudacpu_make_4D<unsigned short>
using int1 = cudacpu_1D_struct<int>;
#define make_int1 cudacpu_make_1D<int>
using int2 = cudacpu_2D_struct<int>;
#define make_int2 cudacpu_make_2D<int>
using int3 = cudacpu_3D_struct<int>;
#define make_int3 cudacpu_make_3D<int>
using int4 = cudacpu_4D_struct<int>;
#define make_int4 cudacpu_make_4D<int>
using uint1 = cudacpu_1D_struct<unsigned int>;
#define make_uint1 cudacpu_make_1D<unsigned int>
using uint2 = cudacpu_2D_struct<unsigned int>;
#define make_uint2 cudacpu_make_2D<unsigned int>
using uint3 = cudacpu_3D_struct<unsigned int>;
#define make_uint3 cudacpu_make_3D<unsigned int>
using uint4 = cudacpu_4D_struct<unsigned int>;
#define make_uint4 cudacpu_make_4D<unsigned int>
using long1 = cudacpu_1D_struct<long>;
#define make_long1 cudacpu_make_1D<long>
using long2 = cudacpu_2D_struct<long>;
#define make_long2 cudacpu_make_2D<long>
using long3 = cudacpu_3D_struct<long>;
#define make_long3 cudacpu_make_3D<long>
using long4 = cudacpu_4D_struct<long>;
#define make_long4 cudacpu_make_4D<long>
using ulong1 = cudacpu_1D_struct<unsigned long>;
#define make_ulong1 cudacpu_make_1D<unsigned long>
using ulong2 = cudacpu_2D_struct<unsigned long>;
#define make_ulong2 cudacpu_make_2D<unsigned long>
using ulong3 = cudacpu_3D_struct<unsigned long>;
#define make_ulong3 cudacpu_make_3D<unsigned long>
using ulong4 = cudacpu_4D_struct<unsigned long>;
#define make_ulong4 cudacpu_make_4D<unsigned long>
using float1 = cudacpu_1D_struct<float>;
#define make_float1 cudacpu_make_1D<float>
using float2 = cudacpu_2D_struct<float>;
#define make_float2 cudacpu_make_2D<float>
using float3 = cudacpu_3D_struct<float>;
#define make_float3 cudacpu_make_3D<float>
using float4 = cudacpu_4D_struct<float>;
#define make_float4 cudacpu_make_4D<float>
using longlong1 = cudacpu_1D_struct<long long>;
#define make_longlong1 cudacpu_make_1D<long long>
using longlong2 = cudacpu_2D_struct<long long>;
#define make_longlong2 cudacpu_make_2D<long long>
using longlong3 = cudacpu_3D_struct<long long>;
#define make_longlong3 cudacpu_make_3D<long long>
using longlong4 = cudacpu_4D_struct<long long>;
#define make_longlong4 cudacpu_make_4D<long long>
using ulonglong1 = cudacpu_1D_struct<unsigned long long>;
#define make_ulonglong1 cudacpu_make_1D<unsigned long long>
using ulonglong2 = cudacpu_2D_struct<unsigned long long>;
#define make_ulonglong2 cudacpu_make_2D<unsigned long long>
using ulonglong3 = cudacpu_3D_struct<unsigned long long>;
#define make_ulonglong3 cudacpu_make_3D<unsigned long long>
using ulonglong4 = cudacpu_4D_struct<unsigned long long>;
#define make_ulonglong4 cudacpu_make_4D<unsigned long long>
using double1 = cudacpu_1D_struct<double>;
#define make_double1 cudacpu_make_1D<double>
using double2 = cudacpu_2D_struct<double>;
#define make_double2 cudacpu_make_2D<double>
using double3 = cudacpu_3D_struct<double>;
#define make_double3 cudacpu_make_3D<double>
using double4 = cudacpu_4D_struct<double>;
#define make_double4 cudacpu_make_4D<double>


//////////////////////////////////////////////////////////////////////////////////////

struct dim3{
    int  x;
    int  y;
    int  z;

    dim3(const int inX = 1, const int inY = 1, const int inZ = 1)
        : x(inX), y(inY), z(inZ){}
};

dim3 make_dim3(const int inX, const int inY = 1, const int inZ = 1){
    return dim3(inX, inY, inZ);
}


#endif
