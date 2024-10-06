#ifndef TEST_SHARED_H
#define TEST_SHARED_H

#include "mc3d_common.h"
#include "gmm.h"
#include "bspline.h"
#include "bspline.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "gmm_param.h"
#include "test_shared.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <iomanip>
#include <memory>

using namespace MC3D_TRECSIM;

Eigen::IOFormat HeavyFormat(Eigen::FullPrecision, 0, ", ", ",\n", "[", "]", "[", "]");

std::vector<std::string> allFrameCameraIds = {
    "camera_1",
    "camera_0",
    "camera_1",
    "camera_0",
    "camera_1",
    "camera_0",
    "camera_1",
    "camera_0",
    "camera_1",
    "camera_0"};

std::vector<double> allFrameOrigTimestamps = {
    1701415521133.655,
    1701415521134.211,
    1701415521156.039,
    1701415521156.4326,
    1701415521178.5227,
    1701415521178.842,
    1701415521200.3083,
    1701415521200.9583,
    1701415521222.0793,
    1701415521222.5564};

std::vector<double> allFrameTimes = {
    1725.5244140625,
    1726.080322265625,
    1747.908447265625,
    1748.302001953125,
    1770.39208984375,
    1770.71142578125,
    1792.177734375,
    1792.82763671875,
    1813.94873046875,
    1814.42578125};

RowMatrix<double> kptsFrame0Person0 = (RowMatrix<double>(17, 3) << 1448.6, 272.41, 0.48682,
                                       1455, 267.32, 0.50049,
                                       1451.8, 264.11, 0.091553,
                                       1472.1, 272.14, 0.83301,
                                       1476.4, 270, 0.17639,
                                       1472.1, 311.79, 0.94238,
                                       1476.4, 303.75, 0.91113,
                                       1475.4, 274.55, 0.8877,
                                       1460.4, 248.04, 0.70459,
                                       1471.1, 233.57, 0.82959,
                                       1450.7, 216.56, 0.61963,
                                       1432.5, 417.86, 0.96094,
                                       1446.4, 415.71, 0.94922,
                                       1369.3, 466.61, 0.94385,
                                       1422.9, 480.8, 0.92139,
                                       1355.4, 542.14, 0.9292,
                                       1450.7, 549.64, 0.91162)
                                          .finished();

RowMatrix<double> kptsFrame1Person0 = (RowMatrix<double>(17, 3) << 1260, 343.93, 0.036987,
                                       1256.8, 336.96, 0.040314,
                                       1266.4, 338.3, 0.0087814,
                                       1238.6, 349.55, 0.74951,
                                       1303.9, 356.25, 0.51807,
                                       1201.1, 408.75, 0.96582,
                                       1318.9, 416.79, 0.96924,
                                       1103.6, 377.14, 0.92139,
                                       1353.2, 355.45, 0.92188,
                                       1047.3, 322.23, 0.84375,
                                       1354.3, 299.46, 0.83398,
                                       1204.3, 586.07, 0.98096,
                                       1277.1, 596.79, 0.98193,
                                       1174.3, 650.89, 0.96631,
                                       1264.3, 703.39, 0.96631,
                                       1180.7, 725.36, 0.9502,
                                       1258.9, 812.14, 0.9502)
                                          .finished();

RowMatrix<double> kptsFrame2Person0 = (RowMatrix<double>(17, 3) << 1442.1, 270.8, 0.061523,
                                       1443.2, 261.96, 0.065613,
                                       1446.4, 262.5, 0.01078,
                                       1431.4, 266.79, 0.69824,
                                       1468.9, 269.46, 0.32397,
                                       1427.1, 298.66, 0.94775,
                                       1462.5, 297.86, 0.94873,
                                       1404.6, 324.91, 0.80566,
                                       1470, 273.21, 0.79199,
                                       1387.5, 361.61, 0.65088,
                                       1472.1, 244.82, 0.61914,
                                       1397.1, 400.71, 0.96289,
                                       1422.9, 403.12, 0.96338,
                                       1353.2, 460.98, 0.94385,
                                       1407.9, 468.21, 0.93896,
                                       1309.3, 518.84, 0.92822,
                                       1437.9, 531.96, 0.92383)
                                          .finished();

RowMatrix<double> kptsFrame3Person0 = (RowMatrix<double>(17, 3) << 1226.8, 310.98, 0.041779,
                                       1225.7, 303.75, 0.050049,
                                       1237.5, 303.75, 0.0068779,
                                       1220.4, 318.21, 0.81982,
                                       1292.1, 324.64, 0.44629,
                                       1183.9, 376.61, 0.9707,
                                       1305, 385.18, 0.96875,
                                       1085.4, 344.73, 0.9375,
                                       1325.4, 319.82, 0.9082,
                                       1006.1, 287.14, 0.87354,
                                       1335, 260.36, 0.81543,
                                       1194.6, 560.36, 0.98291,
                                       1268.6, 568.93, 0.98193,
                                       1168.9, 647.68, 0.96924,
                                       1255.7, 683.57, 0.96631,
                                       1174.3, 683.04, 0.95312,
                                       1261.1, 786.96, 0.94922)
                                          .finished();

RowMatrix<double> kptsFrame4Person0 = (RowMatrix<double>(17, 3) << 1402.5, 248.84, 0.36108,
                                       1405.7, 243.75, 0.25977,
                                       1405.7, 243.21, 0.069031,
                                       1408.9, 250.31, 0.62695,
                                       1427.1, 252.59, 0.29907,
                                       1412.1, 283.12, 0.94141,
                                       1418.6, 278.3, 0.94775,
                                       1391.8, 201.03, 0.875,
                                       1399.3, 200.89, 0.86816,
                                       1382.1, 143.44, 0.83398,
                                       1382.1, 147.19, 0.80908,
                                       1371.4, 392.68, 0.95898,
                                       1376.8, 390.27, 0.96094,
                                       1339.3, 454.29, 0.93701,
                                       1373.6, 454.82, 0.93604,
                                       1283.6, 512.95, 0.91455,
                                       1422.9, 510.27, 0.91406)
                                          .finished();

RowMatrix<double> kptsFrame5Person0 = (RowMatrix<double>(17, 3) << 1212.9, 300, 0.12634,
                                       1212.9, 289.02, 0.12585,
                                       1218.2, 290.89, 0.018509,
                                       1207.5, 295.98, 0.7793,
                                       1275, 300.54, 0.2605,
                                       1177.5, 357.32, 0.96387,
                                       1294.3, 354.64, 0.9292,
                                       1077.3, 337.5, 0.91797,
                                       1281.4, 324.11, 0.62744,
                                       994.82, 278.57, 0.81836,
                                       1239.6, 284.2, 0.46729,
                                       1186.1, 532.5, 0.97949,
                                       1263.2, 534.38, 0.96973,
                                       1166.8, 623.04, 0.9668,
                                       1248.2, 648.21, 0.94385,
                                       1155, 657.86, 0.94531,
                                       1261.1, 769.29, 0.91797)
                                          .finished();

RowMatrix<double> kptsFrame6Person0 = (RowMatrix<double>(17, 3) << 1403.6, 248.17, 0.34717,
                                       1405.7, 242.95, 0.24829,
                                       1405.7, 242.41, 0.065247,
                                       1408.9, 249.24, 0.62109,
                                       1427.1, 251.92, 0.29907,
                                       1412.1, 282.05, 0.94092,
                                       1418.6, 277.5, 0.94775,
                                       1391.8, 199.96, 0.875,
                                       1399.3, 200.36, 0.86768,
                                       1382.1, 143.04, 0.83398,
                                       1382.1, 146.79, 0.80811,
                                       1371.4, 392.14, 0.95898,
                                       1376.8, 390, 0.96094,
                                       1339.3, 453.75, 0.93701,
                                       1373.6, 454.82, 0.93604,
                                       1284.6, 512.95, 0.91504,
                                       1422.9, 510.27, 0.91455)
                                          .finished();

RowMatrix<double> kptsFrame7Person0 = (RowMatrix<double>(17, 3) << 1198.9, 286.61, 0.10486,
                                       1196.8, 279.11, 0.092041,
                                       1206.4, 278.3, 0.019608,
                                       1187.1, 286.61, 0.63379,
                                       1247.1, 285.27, 0.38965,
                                       1165.7, 338.84, 0.95703,
                                       1270.7, 334.02, 0.95312,
                                       1073.6, 323.04, 0.91895,
                                       1292.1, 293.57, 0.85596,
                                       991.61, 258.21, 0.84375,
                                       1281.4, 245.49, 0.7251,
                                       1178.6, 506.25, 0.97363,
                                       1242.9, 510, 0.97168,
                                       1165.7, 586.07, 0.9541,
                                       1239.6, 608.57, 0.94678,
                                       1173.2, 640.18, 0.93213,
                                       1252.5, 718.93, 0.92383)
                                          .finished();

RowMatrix<double> kptsFrame8Person0 = (RowMatrix<double>(17, 3) << 1353.2, 235.98, 0.072266,
                                       1356.4, 228.62, 0.10211,
                                       1356.4, 228.62, 0.015488,
                                       1356.4, 235.58, 0.68701,
                                       1384.3, 237.72, 0.28198,
                                       1342.5, 267.86, 0.93945,
                                       1385.4, 275.89, 0.93604,
                                       1344.6, 257.81, 0.73779,
                                       1398.2, 282.32, 0.67773,
                                       1362.9, 225.27, 0.55762,
                                       1388.6, 240.27, 0.47461,
                                       1312.5, 376.07, 0.94971,
                                       1344.6, 378.21, 0.94727,
                                       1278.2, 438.75, 0.92676,
                                       1312.5, 441.96, 0.91602,
                                       1273.9, 503.04, 0.91211,
                                       1347.9, 487.5, 0.90332)
                                          .finished();

RowMatrix<double> kptsFrame9Person0 = (RowMatrix<double>(17, 3) << 1183.9, 269.46, 0.054413,
                                       1183.9, 263.84, 0.038239,
                                       1191.4, 263.57, 0.013847,
                                       1172.1, 275.89, 0.67627,
                                       1232.1, 270.8, 0.45508,
                                       1147.5, 327.32, 0.96631,
                                       1253.6, 323.57, 0.96777,
                                       1061.2, 301.07, 0.93896,
                                       1308.2, 257.95, 0.93115,
                                       985.18, 238.93, 0.88574,
                                       1338.2, 178.79, 0.86523,
                                       1161.4, 488.3, 0.98242,
                                       1225.7, 489.11, 0.98291,
                                       1157.1, 565.18, 0.96582,
                                       1225.7, 581.25, 0.96582,
                                       1178.6, 633.21, 0.94336,
                                       1241.8, 671.79, 0.94336)
                                          .finished();

std::vector<RowMatrix<double>> allFrameKpts = {
    kptsFrame0Person0,
    kptsFrame1Person0,
    kptsFrame2Person0,
    kptsFrame3Person0,
    kptsFrame4Person0,
    kptsFrame5Person0,
    kptsFrame6Person0,
    kptsFrame7Person0,
    kptsFrame8Person0,
    kptsFrame9Person0};

Camera<double> initCamera0()
{
    IntrinsicMatrix<double> A;
    A << 1.123478282832530113e+03, 0.000000000000000000e+00, 9.624201914371442399e+02, 0.000000000000000000e+00, 1.115995046593396864e+03, 5.875233741561746683e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    DistortionVector<double> d;
    d << -4.098335442199602108e-01, 2.065881602949584117e-01, -1.852675935335392279e-03, -1.274891630199873462e-04, -6.102605512197318421e-02;
    ExtrinsicMatrix<double> P = ExtrinsicMatrix<double>::Identity(4, 4);
    unsigned int height = 1280;
    unsigned int width = 1920;
    Camera<double> camera("camera_0");
    camera.setCalibration(A, d, P, height, width);
    return camera;
};

Camera<double> initCamera1()
{
    IntrinsicMatrix<double> A;
    A << 1.137844469366448493e+03, 0.000000000000000000e+00, 9.258192763436687756e+02, 0.000000000000000000e+00, 1.137868503272385851e+03, 5.874861875982957145e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    DistortionVector<double> d;
    d << -4.122490927804370875e-01, 1.995424107684372617e-01, -6.780183396248970658e-04, 1.457061937740045934e-03, -5.264488616219945710e-02;
    ExtrinsicMatrix<double> P = ExtrinsicMatrix<double>::Identity(4, 4);
    P << 4.894621227523461293e-01, -4.552116929303474668e-01, 7.437803069524321353e-01, -3.875108756319725103e+02, 4.786322175197727513e-01, 8.532140054413278607e-01, 2.072125992088085233e-01, -1.292527896409998505e+02, -7.289293729456041149e-01, 2.545744989944053183e-01, 6.354949202935263886e-01, 1.774314430787018466e+02, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    unsigned int height = 1280;
    unsigned int width = 1920;
    Camera<double> camera("camera_1");
    camera.setCalibration(A, d, P, height, width);
    return camera;
};

std::vector<Frame<double>> initAllFrames()
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();

    std::vector<Frame<double>> frames;

    for (int i = 0; i < allFrameTimes.size(); i++)
    {
        size_t cameraIndex = (allFrameCameraIds.at(i).compare("camera_0") == 0) ? 0 : 1;
        std::vector<RowMatrix<double>> frameKpts = {allFrameKpts.at(i)};
        std::vector<unsigned int> trackerIndices = {0};  // Only one person per frame
        Frame<double> frame(cameraIndex, frameKpts, trackerIndices, allFrameTimes.at(i), allFrameOrigTimestamps.at(i));
        frames.push_back(frame);
    }

    return frames;
};

std::vector<Frame<double>> initAllFramesDouble(double dist = 100)
{
    std::vector<Frame<double>> frames;
    Vector<double> distVector = Vector<double>::Ones(17) * dist;

    for (int i = 0; i < allFrameTimes.size(); i++)
    {
        size_t cameraIndex = (allFrameCameraIds.at(i).compare("camera_0") == 0) ? 0 : 1;
        RowMatrix<double> alteredFrameKpts = allFrameKpts.at(i);

        alteredFrameKpts.col(0) += distVector;
        alteredFrameKpts.col(1) += distVector;

        std::vector<RowMatrix<double>> frameKpts = {allFrameKpts.at(i), alteredFrameKpts};
        std::vector<unsigned int> trackerIndices = {0, 1};  // Two different keypoint sets for the same frame
        Frame<double> frame(cameraIndex, frameKpts, trackerIndices, allFrameTimes.at(i), allFrameOrigTimestamps.at(i));
        frames.push_back(frame);
    }

    return frames;
};
#endif // TEST_SHARED_H