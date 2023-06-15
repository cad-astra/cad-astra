/*
    This file is part of the CAD-ASTRA distribution (git@github.com:cad-astra/cad-astra.git).
    Copyright (c) 2021-2023 imec-Vision Lab, University of Antwerp.

    This program is free software: you can redistribute it and/or modify  
    it under the terms of the GNU General Public License as published by  
    the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "../include/cuda/test_runner.cuh"

#define EXPORT_DEVICE_FUNCTIONS
#include "../include/cuda/project_mesh.cuh"

#include <iostream>

CREATE_TEST_KERNEL(project_point_cone_beam)
{
    float3 point      = {-0.5f, 0.f, 0.5f};
    float3 source     = {-1.f, 0.f, 0.f};
    float3 det_center = { 1.0f, 0.f, 0.f};
    float3 det_u      = { 0.f, 1.f, 0.f};
    float3 det_v      = { 0.f, 0.f, 1.f};
    size_t det_rows = 4;
    size_t det_cols = 4;

    float t;
    float3 det_topleft = det_center - det_u * (det_cols*0.5f) - det_v * (det_rows*0.5f);

    float2 p_my = project_point_cone(point, source, det_topleft, det_u, det_v, t);
    float2 p_ref ={2.f, 4.f};

    // printf("p_ref = (%.2f, %.2f), p_my = (%.2f, %.2f)\n", p_my.x, p_my.y, p_ref.x, p_ref.y);

    ASSERT_EQ_CU(p_my, p_ref)
}

CREATE_TEST_KERNEL(ray_triangle_intersection_simple)
{
    /* Triangle in YZ plane, counterclockwise vertex ordering, i.e., A->B->C:
        C
      / |
    A   |
      \ |
        B
    */
    float3 A = {-1.f, 0.f, 1.f};
    float3 B = {-1.f, 2.f, 0.f};
    float3 C = {-1.f, 2.f, 3.f};

    float3 source = {1.f, 0.f, 0.f};
    float3 det_center = {-2.f, 0.f, 0.f};
    float3 det_u = {0.f, 1.f, 0.f};
    float3 det_v = {0.f,  0.f, 1.f};

    size_t det_rows = 4;
    size_t det_cols = 4;

    float t;
    float3 det_topleft = det_center - det_u * (det_cols*0.5f) - det_v * (det_rows*0.5f);

    float2 A_proj = project_point_cone(A, source, det_topleft, det_u, det_v, t);  // x_A, y_A
    float2 B_proj = project_point_cone(B, source, det_topleft, det_u, det_v, t);  // x_B, y_B
    float2 C_proj = project_point_cone(C, source, det_topleft, det_u, det_v, t);  // x_C, y_C

    int x = 3; int y = 3;
    float2 det_pos = scale(x+0.5f, float2{1.f, 0.f}) + scale(y+0.5f, float2{0.f, 1.f});
    bool intersects = point_in_triangle_2d(A_proj, B_proj, C_proj, det_pos);
    ASSERT_EQ_CU(true, intersects)

    x = 3; y = 5;
    det_pos = scale(x+0.5f, float2{1.f, 0.f}) + scale(y+0.5f, float2{0.f, 1.f});
    intersects = point_in_triangle_2d(A_proj, B_proj, C_proj, det_pos);
    ASSERT_EQ_CU(false, intersects)

    x = 5; y = 3;
    det_pos = scale(x+0.5f, float2{1.f, 0.f}) + scale(y+0.5f, float2{0.f, 1.f});
    intersects = point_in_triangle_2d(A_proj, B_proj, C_proj, det_pos);
    ASSERT_EQ_CU(false, intersects)
}

CREATE_TEST_KERNEL(ray_triangle_intersection_bunny)
{

    float3 A_1 = {-59.078873f, 129.66179f,  75.14452f};
    float3 B_1 = {-61.591663f, 130.77313f,  76.71729f};
    float3 C_1 = {-59.078873f, 130.77313f,  76.6578f };

    float3 A_2 = {-59.078873f, 129.66179f,  75.14452f};
    float3 B_2 = {-59.078873f, 130.77313f,  76.6578f };
    float3 C_2 = {-56.566082f, 129.39423f,  75.14452f};

    float3 source = {-5000.f, 0.f, 0.f};
    float3 det_center = {200.f, 0.f, 0.f};

    float3 det_u = float3{0.f, -1.f, 0.f};
    float3 det_v = float3{0.f, 0.f, -1.f};

    size_t det_rows = 16;
    size_t det_cols = 16;

    float t;
    float3 det_topleft = det_center - det_u * (det_cols*0.5f) - det_v * (det_rows*0.5f);

    float2 A_1_proj = project_point_cone(A_1, source, det_topleft, det_u, det_v, t);  // x_A, y_A
    float2 B_1_proj = project_point_cone(B_1, source, det_topleft, det_u, det_v, t);  // x_B, y_B
    float2 C_1_proj = project_point_cone(C_1, source, det_topleft, det_u, det_v, t);  // x_C, y_C

    float2 A_2_proj = project_point_cone(A_2, source, det_topleft, det_u, det_v, t);  // x_A, y_A
    float2 B_2_proj = project_point_cone(B_2, source, det_topleft, det_u, det_v, t);  // x_B, y_B
    float2 C_2_proj = project_point_cone(C_2, source, det_topleft, det_u, det_v, t);  // x_C, y_C

    float3 det_pixel = {200.00, 137.50, 80.50};
    float2 det_pos = project_point_cone(det_pixel, source, det_topleft, det_u, det_v, t);

    ASSERT_EQ_CU(point_in_triangle_2d(A_1_proj, B_1_proj, C_1_proj, det_pos), point_in_triangle_2d(A_2_proj, B_2_proj, C_2_proj, det_pos))
}

CREATE_TEST_KERNEL(ray_triangle_intersection_cube)
{
    float3 A_1 = {-180.f, -180.f,  180.f};
    float3 B_1 = {-180.f,  180.f, -180.f};
    float3 C_1 = {-180.f, -180.f, -180.f};

    float3 A_2 = {-180.f, -180.f,  180.f};
    float3 B_2 = {-180.f,  180.f,  180.f};
    float3 C_2 = {-180.f,  180.f, -180.f};

    float3 source = {-5000.f, 0.f, 0.f};
    float3 det_center = {200.f, 0.f, 0.f};

    float3 det_u = float3{0.f, -32.f, 0.f};
    float3 det_v = float3{0.f, 0.f, -32.f};

    size_t det_rows = 16;
    size_t det_cols = 16;

    float t;
    float3 det_topleft = det_center - det_u * (det_cols*0.5f) - det_v * (det_rows*0.5f);

    float2 A_1_proj = project_point_cone(A_1, source, det_topleft, det_u, det_v, t);  // x_A, y_A
    float2 B_1_proj = project_point_cone(B_1, source, det_topleft, det_u, det_v, t);  // x_B, y_B
    float2 C_1_proj = project_point_cone(C_1, source, det_topleft, det_u, det_v, t);  // x_C, y_C

    float2 A_2_proj = project_point_cone(A_2, source, det_topleft, det_u, det_v, t);  // x_A, y_A
    float2 B_2_proj = project_point_cone(B_2, source, det_topleft, det_u, det_v, t);  // x_B, y_B
    float2 C_2_proj = project_point_cone(C_2, source, det_topleft, det_u, det_v, t);  // x_C, y_C

    float3 det_pixel = {200.00, -144.00, 144.00};
    float2 det_pos = project_point_cone(det_pixel, source, det_topleft, det_u, det_v, t);

    ASSERT_EQ_CU( point_in_triangle_2d(A_1_proj, B_1_proj, C_1_proj, det_pos),
                 !point_in_triangle_2d(A_2_proj, B_2_proj, C_2_proj, det_pos))
}

void run_all_tests()
{
    CudaKernelTestRunner tr;

    RUN_TEST(tr, project_point_cone_beam);
    RUN_TEST(tr, ray_triangle_intersection_simple);
    RUN_TEST(tr, ray_triangle_intersection_bunny);
    RUN_TEST(tr, ray_triangle_intersection_cube);
}

int main()
{
    std::cerr << "\n";
    std::cerr << "Running tests..." << std::endl;
    std::cerr << "\n";
    run_all_tests();

    return 0;
}