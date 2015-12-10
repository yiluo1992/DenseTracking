#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/cudev/common.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "time.h"
#include "math.h"

#include <vector>
#include <algorithm>

using namespace std;


enum RefImage {LeftRefImage, RightRefImage};

struct CostVolumeParams {

    uint8_t min_disp;
    uint8_t max_disp;
    uint8_t num_disp_layers;
    uint8_t method; // 0 for AD, 1 for ZNCC
    uint8_t win_r;
    RefImage ref_img;

};

struct PrimalDualParams {

    uint32_t num_itr;

    float alpha;
    float beta;
    float epsilon;
    float lambda;
    float aux_theta;
    float aux_theta_gamma;

    /* With preconditoining, we don't need these. */
    float sigma;
    float tau;
    float theta;

};

extern "C"
void runCudaPart(cv::cuda::PtrStepSz<float> data, cv::cuda::PtrStepSz<float> result,int rows, int cols);
void runCudaSet(float* data, float* result, int rows, int cols);

cv::Mat stereoCalcu(int _m, int _n, float* _left_img, float* _right_img, CostVolumeParams _cv_params, PrimalDualParams _pd_params);
void denseTracking(cv::Mat _img1, cv::Mat _img2, cv::Mat _image3D, cv::Mat _cameraMatrx, cv::Mat & _xi);
void computeJacobian(cv::Mat _img1, cv::Mat _img2, cv::Mat _image3D, cv::Mat _cameraMatrx, cv::Mat xi, cv::Mat & _img_residual, cv::Mat _W, cv::Mat & _hessian, cv::Mat & _sdpara, double & _error, float & _sigma);

int main()
{
    cv::Mat Img_rgb_left;
    cv::Mat Img_rgb_right;
    cv::Mat Img_rgb_left1;
    cv::Mat Img_rgb_right1;

    cv::VideoCapture VCLeft;
    VCLeft.open("/home/roy/Data/vivo2/left.avi");
    //VCLeft.open("/home/roy/Data/vivo3/left.avi");
    VCLeft.read(Img_rgb_left);
    for(int i = 0; i < 50; i++)
        VCLeft.read(Img_rgb_left1);
    VCLeft.release();

    cv::VideoCapture VCRight;
    VCRight.open("/home/roy/Data/vivo2/right.avi");
    //VCRight.open("/home/roy/Data/vivo3/right.avi");
    VCRight.read(Img_rgb_right);
    for(int i = 0; i < 50; i++)
        VCRight.read(Img_rgb_right1);
    VCRight.release();

//    cv::Mat Img_rgb_left = cv::imread("/home/roy/Data/vivo2/unrec_left.jpg");
//    cv::Mat Img_rgb_right = cv::imread("/home/roy/Data/vivo2/unrec_right.jpg");
    cv::Mat Img_gray_left;
    cv::Mat Img_gray_right;
    cvtColor(Img_rgb_left, Img_gray_left, CV_RGB2GRAY);
    cvtColor(Img_rgb_right, Img_gray_right, CV_RGB2GRAY);

    cv::Mat Img_gray_left1;
    cv::Mat Img_gray_right1;
    cvtColor(Img_rgb_left1, Img_gray_left1, CV_RGB2GRAY);
    cvtColor(Img_rgb_right1, Img_gray_right1, CV_RGB2GRAY);

    int rows = Img_rgb_left.rows;
    int cols = Img_rgb_left.cols;

    cv::Size Img_size(cols, rows);

    // vivo2 true!!!
    cv::Mat K_right = (cv::Mat_<float>(3,3) << 751.706129, 0.000000, 338.665467,
                   0.000000, 766.315580, 257.986032,
                   0.000000, 0.000000, 1.000000);
    cv::Mat Distortion_right = (cv::Mat_<float>(1,4) << -0.328424, 0.856059, 0.003430, 0.000248);

    cv::Mat K_left = (cv::Mat_<float>(3,3) << 752.737796, 0.000000, 263.657845,
                  0.000000, 765.331797, 245.883296,
                  0.000000, 0.000000, 1.000000);
    cv::Mat Distortion_left = (cv::Mat_<float>(1,4) << -0.332222, 0.708196, 0.003904, 0.008043);

    cv::Mat R_StereoCamera = (cv::Mat_<double>(3,3) << 0.9999, 0.0069, -0.0121,
                          -0.0068, 1.0000, 0.0055,
                          0.0121, -0.0054, 0.9999);
    R_StereoCamera = R_StereoCamera.inv();

    cv::Mat T_StereoCamera = (cv::Mat_<double>(3,1) << -5.3754, -0.0716, -0.1877);

        // vivo3
//    cv::Mat K_left = (cv::Mat_<float>(3,3) << 391.656525, 0.000000, 165.964371,
//                      0.000000, 426.835144, 154.498138,
//                      0.000000, 0.000000, 1.000000);
//    cv::Mat Distortion_left = (cv::Mat_<float>(1,4) << -0.196312, 0.129540, 0.004356, 0.006236);

//    cv::Mat K_right = (cv::Mat_<float>(3,3) << 390.376862, 0.000000, 190.896454,
//                       0.000000, 426.228882, 145.071411,
//                       0.000000, 0.000000, 1.000000);
//    cv::Mat Distortion_right = (cv::Mat_<float>(1,4) << -0.205824, 0.186125, 0.015374, 0.003660);

//    cv::Mat R_StereoCamera = (cv::Mat_<double>(3,3) << 0.999999, -0.001045, -0.000000,
//                              0.001045, 0.999999, -0.000000,
//                              0.000000, 0.000000, 1.000000);
//    cv::Mat T_StereoCamera = (cv::Mat_<double>(3,1) << -5.520739, -0.031516, -0.051285);

    //cv::Mat Img_rgb_left_undistor;
    //cv::Mat K_left_undistor;
    //undistort(Img_rgb_left, Img_rgb_left_undistor, K_left, Distortion_left, K_left_undistor);

    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect roi1, roi2;

    cv::stereoRectify(K_left, Distortion_left, K_right, Distortion_right,  Img_size, R_StereoCamera, T_StereoCamera, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0, Img_size, &roi1, &roi2);
    //cout << Q;
    //cout << P1 << endl;
    //cout << P2 << endl;

    cv::Mat map11, map12, map21, map22;
    cv::initUndistortRectifyMap(K_left, Distortion_left, R1, P1, Img_size, CV_16SC2, map11, map12);
    cv::initUndistortRectifyMap(K_right, Distortion_right, R2, P2, Img_size, CV_16SC2, map21, map22);

    cv::Mat img_rec_left, img_rec_right;
    cv::remap(Img_gray_left, img_rec_left, map11, map12, cv::INTER_LINEAR);
    cv::remap(Img_gray_right, img_rec_right, map21, map22, cv::INTER_LINEAR);
    img_rec_left.convertTo(img_rec_left, CV_32F);
    img_rec_right.convertTo(img_rec_right, CV_32F);

    cv::Mat img_rec_left1, img_rec_right1;
    cv::remap(Img_gray_left1, img_rec_left1, map11, map12, cv::INTER_LINEAR);
    cv::remap(Img_gray_right1, img_rec_right1, map21, map22, cv::INTER_LINEAR);
    img_rec_left1.convertTo(img_rec_left1, CV_32F);
    img_rec_right1.convertTo(img_rec_right1, CV_32F);

    //cv::resize(img_left_gray, img_left_gray, Size(img_left_gray.cols/2, img_left_gray.rows/2));
    //cv::resize(img_right_gray, img_right_gray, Size(img_right_gray.cols/2, img_right_gray.rows/2));

    CostVolumeParams cv_params;
    cv_params.min_disp = 0; // 0
    cv_params.max_disp = 64; // 64
    cv_params.method = 1;
    cv_params.win_r = 10;
    cv_params.ref_img = LeftRefImage;

    PrimalDualParams pd_params;
//    pd_params.num_itr = 150; // 500
//    pd_params.alpha = 0.05; // 10.0
//    pd_params.beta = 1.0; // 1.0
//    pd_params.epsilon = 0.1; // 0.1
//    pd_params.lambda = 1e-2; // 1e-3
//    pd_params.aux_theta = 10; // 10
//    pd_params.aux_theta_gamma = 1e-6; // 1e-6
    pd_params.num_itr = 200; // 500
    pd_params.alpha = 0.1; // 10.0
    pd_params.beta = 1.0; // 1.0
    pd_params.epsilon = 0.1; // 0.1
    pd_params.lambda = 1e-2; // 1e-3
    pd_params.aux_theta = 10; // 10
    pd_params.aux_theta_gamma = 1e-6; // 1e-6


    cv::Mat result = stereoCalcu(img_rec_left.rows, img_rec_left.cols, (float*)img_rec_left.data, (float*)img_rec_right.data, cv_params, pd_params);
    // convert for [0,1] to [min_d, max_d]
    result.convertTo(result, CV_32F, cv_params.max_disp);

    //---------rectify the rgb imag
    cv::remap(Img_rgb_left, Img_rgb_left, map11, map12, cv::INTER_LINEAR);

    // ------------------------------------
    // -----Triangulate Using Disparity-----
    // ------------------------------------

    cv::Mat image3D;
    cv::reprojectImageTo3D(result, image3D, Q);

    // ---------------------------
    // ------------------------------------
    // -----From xi to Rigid Transformation Matrix-----
    // ------------------------------------
    cv::Mat r_vec = (cv::Mat_<float>(3,1) << 0.01, 0.01, 0.01);
    cv::Mat t_vec = (cv::Mat_<float>(3,1) << 0.3, 0.5, 0.2);
    cv::Mat r_mat;
    cv::Rodrigues(r_vec, r_mat);
    cv::Mat trans_mat(4,4,CV_32F);
    trans_mat.at<float>(0,0) = r_mat.at<float>(0,0);
    trans_mat.at<float>(0,1) = r_mat.at<float>(0,1);
    trans_mat.at<float>(0,2) = r_mat.at<float>(0,2);
    trans_mat.at<float>(1,0) = r_mat.at<float>(1,0);
    trans_mat.at<float>(1,1) = r_mat.at<float>(1,1);
    trans_mat.at<float>(1,2) = r_mat.at<float>(1,2);
    trans_mat.at<float>(2,0) = r_mat.at<float>(2,0);
    trans_mat.at<float>(2,1) = r_mat.at<float>(2,1);
    trans_mat.at<float>(2,2) = r_mat.at<float>(2,2);
    trans_mat.at<float>(0,3) = t_vec.at<float>(0);
    trans_mat.at<float>(1,3) = t_vec.at<float>(1);
    trans_mat.at<float>(2,3) = t_vec.at<float>(2);
    trans_mat.at<float>(3,0) = 0.0;
    trans_mat.at<float>(3,1) = 0.0;
    trans_mat.at<float>(3,2) = 0.0;
    trans_mat.at<float>(3,3) = 1.0;
    //cout << r_mat;
    cout << trans_mat << endl;
    cv::Mat cameraMatrx;
    P1.convertTo(cameraMatrx, CV_32F);

    // ------------------------------------
    // ----Get I2-----
    // ------------------------------------
    cv::Mat_<float> xmap(image3D.rows,image3D.cols,-9999.9);//9999.9's are to guarantee that pixels are invalid
    cv::Mat_<float> ymap(image3D.rows,image3D.cols,-9999.9);//9999.9's are to guarantee that pixels are invalid

    for(int x = 0; x < image3D.cols; x++)
    {
        for(int y = 0; y < image3D.rows; y++)
        {
            cv::Vec3f point_vec = image3D.at<cv::Vec3f>(y,x);
            //cout << point_vec;
            if(!cvIsInf(point_vec.val[0]) && !cvIsInf(point_vec.val[1]) && !cvIsInf(point_vec.val[2]))
            {
                cv::Mat homo_point_mat(4, 1, CV_32F);
                homo_point_mat.at<float>(0) = point_vec.val[0];
                homo_point_mat.at<float>(1) = point_vec.val[1];
                homo_point_mat.at<float>(2) = point_vec.val[2];
                homo_point_mat.at<float>(3) = 1;

                // add noise
                srand (time(0));
                //homo_point_mat.at<float>(0) += (rand()%10000)/10000.0;
                //homo_point_mat.at<float>(1) += (rand()%10000)/10000.0;
                //homo_point_mat.at<float>(2) += (rand()%10000)/10000.0;

                cv::Mat point_projected = cameraMatrx*trans_mat*homo_point_mat;
                point_projected.convertTo(point_projected, CV_32F, 1/point_projected.at<float>(2));
                //cout << point_projected;
                float dst_x = point_projected.rowRange(0,2).at<float>(0);
                float dst_y = point_projected.rowRange(0,2).at<float>(1);
                if(dst_y < rows && dst_x < cols && dst_y >= 0 && dst_x > 0)
                {
                    xmap.at<float>(dst_y,dst_x) = x;
                    ymap.at<float>(dst_y,dst_x) = y;
                }
            }
        }
    }
    cv::Mat img_projected;
    cv::remap(img_rec_left, img_projected, xmap, ymap, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    img_projected.convertTo(img_projected, CV_8U);
    cv::imshow("projected", img_projected);
    img_projected.convertTo(img_projected, CV_32F);
    cv::waitKey(0);
    // -----------------------------------
    // -----Dense Tracking-----------------
    // ------------------------------------
    cv::Mat xi = (cv::Mat_<float>(6,1) << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    //img_rec_left.convertTo(img_rec_left1, CV_32F, 1.0/255);
    //img_projected.convertTo(img_projected, CV_32F, 1.0/255);
    //img_rec_left1.convertTo(img_rec_left1, CV_32F, 1.0/255);
    denseTracking(img_rec_left, img_projected, image3D, P1, xi);
    //denseTracking(img_rec_left, img_rec_left1, image3D, P1, xi);

    //cout << xi.rowRange(3,6);

    // Cut the image due to the invalid point in estimating disparity
    // image3D = image3D.rowRange(9, rows-9).clone();
    // image3D = image3D.colRange(18 + 64, cols-9).clone();
    // img_rec_left = img_rec_left.rowRange(9, rows-9).clone();
    // img_rec_left = img_rec_left.colRange(18 + 64, cols-9).clone();
    // img_rec_left1 = img_rec_left1.rowRange(9, rows-9).clone();
    // img_rec_left1 = img_rec_left1.colRange(18 + 64, cols-9).clone();



    // Normalize the disparity img to show
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    result.convertTo(result, CV_8U);

    //img_left_reprojected.convertTo(img_left_reprojected, CV_8U);
    //cv::imshow("left_reprojected", img_left_reprojected);
    cv::imshow("disparity", result);
    //cv::imshow("left", Img_rgb_left);
    //cv::imshow("residual", img_residual);
    cv::waitKey(0);

    return 0;

}

void denseTracking(cv::Mat _img1, cv::Mat _img2, cv::Mat _image3D, cv::Mat _cameraMatrx, cv::Mat &_xi)
{
    /*
        Input: I1(m*n), I2, I1 3D point/I1_DepthMap, CameraMatrix, xi
        Output: xi
        1. warp the I2 image to get I2*
        2. Computer residual image I2*-I1, Ir
        3. Computer per-pixel scale according to Ir
        4. Computer per-pixel Weight, W
        5. Calculate Jacobian Matrix JI (1*2), the gradient image of I2*, I2*_x, I2*_y
        6. Compute the Jacobian Matrix Jw (2*6)
        7. J = JI*Jw (1*6)
        8. Sum the J according to all pixel in I1 and weighted by W
        9. Computer delta = inv(J.t()*J*W(xi))*J.t()*W(xi)*Ir(xi)
        10. incorporate increment
    */

    cv::Mat delta;
    cv::Mat hessian;
    cv::Mat W;
    cv::Mat sdpara;
    cv::Mat img_residual;
    cv::Mat img_residual_start;
    double error = 0;
    float sigma = 1;

    int max_iter = 1000;
    clock_t start, finish;

    int k = 0;
    bool converged = false;
    while(k < max_iter && !converged)
    {
        start = clock();

        computeJacobian(_img1, _img2, _image3D, _cameraMatrx, _xi, img_residual, W, hessian, sdpara, error, sigma);

        cv::solve(hessian, sdpara, delta, cv::DECOMP_CHOLESKY);
        //cout << hessian.inv() << endl;
        //cout << sdpara << endl;
        //delta = hessian.inv()*sdpara;
        //cout << delta << endl;
        //cout << hessian*delta << endl;
        //cout << hessian.inv()*hessian << endl;

        // ------------------------------------
        // -----From xi to Rigid Transformation Matrix-----
        // ------------------------------------
        cv::Mat r_mat;
        cv::Rodrigues(_xi.rowRange(3,6), r_mat);
        cv::Mat trans_mat_original(4,4,CV_32F);
        trans_mat_original.at<float>(0,0) = r_mat.at<float>(0,0);
        trans_mat_original.at<float>(0,1) = r_mat.at<float>(0,1);
        trans_mat_original.at<float>(0,2) = r_mat.at<float>(0,2);
        trans_mat_original.at<float>(1,0) = r_mat.at<float>(1,0);
        trans_mat_original.at<float>(1,1) = r_mat.at<float>(1,1);
        trans_mat_original.at<float>(1,2) = r_mat.at<float>(1,2);
        trans_mat_original.at<float>(2,0) = r_mat.at<float>(2,0);
        trans_mat_original.at<float>(2,1) = r_mat.at<float>(2,1);
        trans_mat_original.at<float>(2,2) = r_mat.at<float>(2,2);
        trans_mat_original.at<float>(0,3) = _xi.at<float>(0);
        trans_mat_original.at<float>(1,3) = _xi.at<float>(1);
        trans_mat_original.at<float>(2,3) = _xi.at<float>(2);
        trans_mat_original.at<float>(3,0) = 0.0;
        trans_mat_original.at<float>(3,1) = 0.0;
        trans_mat_original.at<float>(3,2) = 0.0;
        trans_mat_original.at<float>(3,3) = 1.0;

        //cout << trans_mat_original << endl;

        cv::Rodrigues(delta.rowRange(3,6), r_mat);
        cv::Mat trans_mat_increment(4,4,CV_32F);
        trans_mat_increment.at<float>(0,0) = r_mat.at<float>(0,0);
        trans_mat_increment.at<float>(0,1) = r_mat.at<float>(0,1);
        trans_mat_increment.at<float>(0,2) = r_mat.at<float>(0,2);
        trans_mat_increment.at<float>(1,0) = r_mat.at<float>(1,0);
        trans_mat_increment.at<float>(1,1) = r_mat.at<float>(1,1);
        trans_mat_increment.at<float>(1,2) = r_mat.at<float>(1,2);
        trans_mat_increment.at<float>(2,0) = r_mat.at<float>(2,0);
        trans_mat_increment.at<float>(2,1) = r_mat.at<float>(2,1);
        trans_mat_increment.at<float>(2,2) = r_mat.at<float>(2,2);
        trans_mat_increment.at<float>(0,3) = delta.at<float>(0);
        trans_mat_increment.at<float>(1,3) = delta.at<float>(1);
        trans_mat_increment.at<float>(2,3) = delta.at<float>(2);
        trans_mat_increment.at<float>(3,0) = 0.0;
        trans_mat_increment.at<float>(3,1) = 0.0;
        trans_mat_increment.at<float>(3,2) = 0.0;
        trans_mat_increment.at<float>(3,3) = 1.0;

        //cout << trans_mat_increment << endl;

        cv::Mat trans_mat_result = trans_mat_original*trans_mat_increment;
        //cv::Mat trans_mat_result = trans_mat_increment*trans_mat_original;
        cv::Mat r_vec_result;
        cv::Rodrigues(trans_mat_result.rowRange(0,3).colRange(0,3), r_vec_result);

        //cout << r_vec_result << endl;

        _xi.at<float>(0) = trans_mat_result.at<float>(0, 3);
        _xi.at<float>(1) = trans_mat_result.at<float>(1, 3);
        _xi.at<float>(2) = trans_mat_result.at<float>(2, 3);
        _xi.at<float>(3) = r_vec_result.at<float>(0);
        _xi.at<float>(4) = r_vec_result.at<float>(1);
        _xi.at<float>(5) = r_vec_result.at<float>(2);

//        if(k%10 == 0)
//        {
//            cout << "k: " << k << " Error: "<< error << endl;
//            cout << "xi: " << _xi << endl;
//        }
        cout << "k: " << k << " Error: "<< error << endl;
        cout << "xi: " << _xi << endl;

        if(error < 10)
            sigma = 1;

        k++;

        finish = clock();
        double duration = (double)(finish - start) / CLOCKS_PER_SEC;
        //printf("This take %f tracking one round. \n", duration);

        //converged = true;
        if(k == 1)
            img_residual_start = img_residual.clone();
    }
    // show image residual
    cv::Mat img_residual_show;
    cv::Mat img_residual_start_show;
    //cv::normalize(img_residual, img_residual_show, 0, 255, cv::NORM_MINMAX);
    img_residual.convertTo(img_residual_show, CV_8U);
    img_residual_start.convertTo(img_residual_start_show, CV_8U);
    cv::imshow("residual", img_residual_show);
    cv::imshow("residual start", img_residual_start_show);
    //cv::waitKey(0);
}

void computeJacobian(cv::Mat _img1, cv::Mat _img2, cv::Mat _image3D, cv::Mat _cameraMatrx, cv::Mat xi, cv::Mat & _img_residual,cv::Mat _W, cv::Mat & _hessian, cv::Mat & _sdpara, double & _error, float & _sigma )
{
    int edge1 = 20;
    int edge2 = 64;

    int rows = _image3D.rows;
    int cols = _image3D.cols;

    _cameraMatrx.convertTo(_cameraMatrx, CV_32F);
    // ------------------------------------
    // -----From xi to Rigid Transformation Matrix-----
    // ------------------------------------
    cv::Mat r_mat;
    cv::Rodrigues(xi.rowRange(3,6), r_mat);
    cv::Mat trans_mat(4,4,CV_32F);
    trans_mat.at<float>(0,0) = r_mat.at<float>(0,0);
    trans_mat.at<float>(0,1) = r_mat.at<float>(0,1);
    trans_mat.at<float>(0,2) = r_mat.at<float>(0,2);
    trans_mat.at<float>(1,0) = r_mat.at<float>(1,0);
    trans_mat.at<float>(1,1) = r_mat.at<float>(1,1);
    trans_mat.at<float>(1,2) = r_mat.at<float>(1,2);
    trans_mat.at<float>(2,0) = r_mat.at<float>(2,0);
    trans_mat.at<float>(2,1) = r_mat.at<float>(2,1);
    trans_mat.at<float>(2,2) = r_mat.at<float>(2,2);
    trans_mat.at<float>(0,3) = xi.at<float>(0);
    trans_mat.at<float>(1,3) = xi.at<float>(1);
    trans_mat.at<float>(2,3) = xi.at<float>(2);
    trans_mat.at<float>(3,0) = 0.0;
    trans_mat.at<float>(3,1) = 0.0;
    trans_mat.at<float>(3,2) = 0.0;
    trans_mat.at<float>(3,3) = 1.0;
    //cout << r_mat;
    //cout << trans_mat;

    // ------------------------------------
    // -----Warp I2-----
    // ------------------------------------
    cv::Mat_<float> xmap(_image3D.rows,_image3D.cols,-9999.9);//9999.9's are to guarantee that pixels are invalid
    cv::Mat_<float> ymap(_image3D.rows,_image3D.cols,-9999.9);//9999.9's are to guarantee that pixels are invalid


    for(int x = 0; x < _image3D.cols; x++)
    {
        for(int y = 0; y < _image3D.rows; y++)
        {
            cv::Vec3f point_vec = _image3D.at<cv::Vec3f>(y,x);
            //cout << point_vec;
            if(!cvIsInf(point_vec.val[0]) && !cvIsInf(point_vec.val[1]) && !cvIsInf(point_vec.val[2]))
            {
                cv::Mat homo_point_mat(4, 1, CV_32F);
                homo_point_mat.at<float>(0) = point_vec.val[0];
                homo_point_mat.at<float>(1) = point_vec.val[1];
                homo_point_mat.at<float>(2) = point_vec.val[2];
                homo_point_mat.at<float>(3) = 1;
                cv::Mat point_projected =_cameraMatrx*trans_mat*homo_point_mat;
                //point_projected.convertTo(point_projected, CV_32S, 1/point_projected.at<float>(2));
                point_projected.convertTo(point_projected, CV_32F, 1/point_projected.at<float>(2));
                //cout << point_projected;
                float src_x = point_projected.rowRange(0,2).at<float>(0);
                float src_y = point_projected.rowRange(0,2).at<float>(1);
                xmap.at<float>(y,x) = src_x;
                ymap.at<float>(y,x) = src_y;
            }
        }
    }
    cv::Mat img_warped;
    cv::Mat img_warped_show;
    cv::remap(_img2, img_warped, xmap, ymap, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    //img_warped.convertTo(img_warped_show, CV_8U);
    //cv::imshow("warped", img_warped_show);
    //cv::waitKey(0);

    // ------------------------------------
    // ----- Compute residual image-----
    // ------------------------------------
    cv::Mat img_residual(rows, cols, CV_32F, cv::Scalar(0));
    cv::Mat img_residual_show;
    for(int x = edge1+edge2; x < _image3D.cols - edge1; x++)
    {
        for(int y = edge1; y < _image3D.rows - edge1; y++)
        {
            img_residual.at<float>(y,x) = img_warped.at<float>(y,x) - _img1.at<float>(y,x);
        }
    }
    //cv::normalize(img_residual, img_residual_show, 0, 255, cv::NORM_MINMAX);
    //img_residual.convertTo(img_residual_show, CV_8U);
    //cv::imshow("residual", img_residual_show);
    //cv::waitKey(0);

    //img_residual.convertTo(img_residual, CV_32F, 1.0/255);
    _img_residual = img_residual.clone();

    // ------------------------------------
    // -----Compute sigma and tulkey weight image, sum weighted error-----
    // ------------------------------------
    cv::Mat img_weight(rows, cols, CV_32F, cv::Scalar(0));
    //float sigma_pre = _sigma;
    float sigma_now = 0;
    vector<float> sigmaArray;
    size_t n;
    for(int x = edge1+edge2; x < _image3D.cols - edge1; x++)
    {
        for(int y = edge1; y < _image3D.rows - edge1; y++)
        {
            sigmaArray.push_back(img_residual.at<float>(y,x));
        }
    }
    // find median one
    n = sigmaArray.size()/2;
    nth_element(sigmaArray.begin(), sigmaArray.begin()+n, sigmaArray.end());
    float sigma_temp = sigmaArray[n];
    sigmaArray.clear();
    for(int x = edge1+edge2; x < _image3D.cols - edge1; x++)
    {
        for(int y = edge1; y < _image3D.rows - edge1; y++)
        {
            sigmaArray.push_back(abs(img_residual.at<float>(y,x) - sigma_temp));
        }
    }
    // find median two
    n = sigmaArray.size()/2;
    nth_element(sigmaArray.begin(), sigmaArray.begin()+n, sigmaArray.end());
    sigma_now = sigmaArray[n] * (1.4826);
    sigmaArray.clear();


    float error = 0.0;
    float b = 4.6851;
    // weight image
    for(int x = 9 + 64; x < _image3D.cols - 9; x++)
    {
        for(int y = 9; y < _image3D.rows - 9; y++)
        {
            float temp = img_residual.at<float>(y,x);
            float temp_x = temp/sigma_now;
            if(abs(temp_x) <= b )
                img_weight.at<float>(y,x) = pow((1 - pow((temp_x/b),2)),2);
            else
                img_weight.at<float>(y,x) = 0;
            error += abs(img_residual.at<float>(y,x)*img_weight.at<float>(y,x));

            img_weight.at<float>(y,x) = 1;
            //cout << temp << " " << sigma_now << " " << nu << " " <<((nu + 1) / (nu + ((temp*temp) / (sigma_now*sigma_now)))) <<  endl;
            //cout << img_residual.at<float>(y,x) << " " << img_weight.at<float>(y,x)<< " " << error << " \n";
            //cv::waitKey(0);

        }
    }
    //cv::Mat img_weight_show;
    //cv::normalize(img_weight, img_weight_show, 0, 255, cv::NORM_MINMAX);
    //img_weight_show.convertTo(img_weight_show, CV_8U);
    //cv::imshow("weight", img_weight_show);

    _W = img_weight.clone();
    _error = (double)error;

    //    // ------------------------------------
    //    // -----Compute sigma and weight image, sum weighted error-----
    //    // ------------------------------------
    //    cv::Mat img_weight(rows, cols, CV_32F, cv::Scalar(0));
    //    float sigma_pre = 1;
    //    float sigma_now = 0;
    //    float nu = 5;
    //    int count = 0;
    //    for(int x = 9 + 64; x < _image3D.cols - 9; x++)
    //    {
    //        for(int y = 9; y < _image3D.rows - 9; y++)
    //        {

    //            float temp = img_residual.at<float>(y,x);
    //            sigma_now += ((temp*temp)*(nu+1)) / (nu+((temp*temp)/(sigma_pre*sigma_pre)));
    //            count++;
    //        }
    //    }
    //    sigma_now = sigma_now / count;

    //    float error = 0.0;
    //    // weight image
    //    for(int x = 9 + 64; x < _image3D.cols - 9; x++)
    //    {
    //        for(int y = 9; y < _image3D.rows - 9; y++)
    //        {
    //            float temp = img_residual.at<float>(y,x);
    //            if(sigma_now == 0)
    //                img_weight.at<float>(y,x) = 1;
    //            else
    //                img_weight.at<float>(y,x) = ((nu + 1) / (nu + ((temp*temp) / (sigma_now*sigma_now))));

    //            img_weight.at<float>(y,x) = 1;
    //            error += abs(img_residual.at<float>(y,x)*img_weight.at<float>(y,x));

    //            //cout << temp << " " << sigma_now << " " << nu << " " <<((nu + 1) / (nu + ((temp*temp) / (sigma_now*sigma_now)))) <<  endl;
    //            //cout << img_residual.at<float>(y,x) << " " << img_weight.at<float>(y,x)<< " " << error << " \n";
    //            //cv::waitKey(0);

    //        }
    //    }
    //    //cv::Mat img_weight_show;
    //    //cv::normalize(img_weight, img_weight_show, 0, 255, cv::NORM_MINMAX);
    //    //img_weight_show.convertTo(img_weight_show, CV_8U);
    //    //cv::imshow("weight", img_weight_show);

    //    _W = img_weight.clone();
    //    _error = (double)error;


    // ------------------------------------
    // -----Compute JI image-----
    // ------------------------------------
    cv::Mat img_x_grad(rows, cols, CV_32F, cv::Scalar(0));
    cv::Mat img_y_grad(rows, cols, CV_32F, cv::Scalar(0));
    for(int x = edge1+edge2; x < _image3D.cols - edge1; x++)
    {
        for(int y = edge1; y < _image3D.rows - edge1; y++)
        {
            img_x_grad.at<float>(y,x) = (img_warped.at<float>(y,x+1) - img_warped.at<float>(y,x-1))/2;
            img_y_grad.at<float>(y,x) = (img_warped.at<float>(y+1,x) - img_warped.at<float>(y-1,x))/2;
        }
    }
//    cv::Mat img_x_grad_show;
//    cv::Mat img_y_grad_show;
//    cv::normalize(img_x_grad, img_x_grad_show, 0, 255, cv::NORM_MINMAX);
//    cv::normalize(img_y_grad, img_y_grad_show, 0, 255, cv::NORM_MINMAX);
//    img_x_grad_show.convertTo(img_x_grad_show, CV_8U);
//    img_y_grad_show.convertTo(img_y_grad_show, CV_8U);
//    cv::imshow("x_gradient", img_x_grad_show);
//    cv::imshow("y_gradient", img_y_grad_show);

    //cout << img_x_grad.row(200);


    // ------------------------------------
    // -----Compute Jw image-----
    // ------------------------------------
    float focalx = _cameraMatrx.at<float>(0,0);
    float focaly = _cameraMatrx.at<float>(1,1);
    cv::Mat img_Jw(rows*2, cols*6, CV_32F, cv::Scalar(0));
    for(int x = edge1+edge2; x < _image3D.cols - edge1; x++)
    {
        for(int y = edge1; y < _image3D.rows - edge1; y++)
        {
            cv::Vec3f point_vec = _image3D.at<cv::Vec3f>(y,x);
            float x_cor = point_vec.val[0];
            float y_cor = point_vec.val[1];
            float z_cor = point_vec.val[2];
            img_Jw.at<float>((y)*2 + 0, (x)*6 + 0) = focalx/z_cor;
            img_Jw.at<float>((y)*2 + 0, (x)*6 + 1) = 0.0;
            img_Jw.at<float>((y)*2 + 0, (x)*6 + 2) = (-1)*((focalx*x_cor)/(z_cor*z_cor));
            img_Jw.at<float>((y)*2 + 0, (x)*6 + 3) = (-1)*((focalx*(x_cor*y_cor))/(z_cor*z_cor));
            img_Jw.at<float>((y)*2 + 0, (x)*6 + 4) = (focalx*(1 + ((x_cor*x_cor)/(z_cor*z_cor))));
            img_Jw.at<float>((y)*2 + 0, (x)*6 + 5) = (-1)*(focalx*(y_cor/z_cor));

            img_Jw.at<float>((y)*2 + 1, (x)*6 + 0) = 0.0;
            img_Jw.at<float>((y)*2 + 1, (x)*6 + 1) = focaly/z_cor;
            img_Jw.at<float>((y)*2 + 1, (x)*6 + 2) = (-1)*((focaly*y_cor)/(z_cor*z_cor));
            img_Jw.at<float>((y)*2 + 1, (x)*6 + 3) = (-1)*(focaly*(1 + ((y_cor*y_cor)/(z_cor*z_cor))));
            img_Jw.at<float>((y)*2 + 1, (x)*6 + 4) = ((focaly*(x_cor*y_cor))/(z_cor*z_cor));
            img_Jw.at<float>((y)*2 + 1, (x)*6 + 5) = (focaly*(x_cor/z_cor));

        }
    }

    // ------------------------------------
    // -----Compute J image and sum to J matrix-----
    // ------------------------------------
    cv::Mat img_J(rows, cols*6, CV_32F, cv::Scalar(0));
    //cv::Mat J_mat(1, 6, CV_32F, cv::Scalar(0));
    for(int x = edge1+edge2; x < _image3D.cols - edge1; x++)
    {
        for(int y = edge1; y < _image3D.rows - edge1; y++)
        {
            float weight = sqrt(img_weight.at<float>(y,x));
            float x_grad = img_x_grad.at<float>(y,x);
            float y_grad = img_y_grad.at<float>(y,x);
            //float weight = img_weight.at<float>(y,x);
            //cout << x_grad << " " << y_grad << endl;

            img_J.at<float>((y) , (x)*6 + 0) = x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 0) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 0);
            img_J.at<float>((y) , (x)*6 + 1) = x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 1) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 1);
            img_J.at<float>((y) , (x)*6 + 2) = x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 2) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 2);
            img_J.at<float>((y) , (x)*6 + 3) = x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 3) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 3);
            img_J.at<float>((y) , (x)*6 + 4) = x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 4) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 4);
            img_J.at<float>((y) , (x)*6 + 5) = x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 5) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 5);

//            J_mat.at<float>(0) += weight*(x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 0) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 0));
//            J_mat.at<float>(1) += weight*(x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 1) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 1));
//            J_mat.at<float>(2) += weight*(x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 2) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 2));
//            J_mat.at<float>(3) += weight*(x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 3) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 3));
//            J_mat.at<float>(4) += weight*(x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 4) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 4));
//            J_mat.at<float>(5) += weight*(x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 5) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 5));
        }
    }
    //cout << "\nJ is " << J_mat << endl;
    //_J = J_mat.clone();

    // ------------------------------------
    // -----Compute J image Seperately-----
    // ------------------------------------
//    cv::Mat img_J0(rows, cols, CV_32F, cv::Scalar(0));
//    cv::Mat img_J1(rows, cols, CV_32F, cv::Scalar(0));
//    cv::Mat img_J2(rows, cols, CV_32F, cv::Scalar(0));
//    cv::Mat img_J3(rows, cols, CV_32F, cv::Scalar(0));
//    cv::Mat img_J4(rows, cols, CV_32F, cv::Scalar(0));
//    cv::Mat img_J5(rows, cols, CV_32F, cv::Scalar(0));
//    for(int x = 9 + 64; x < _image3D.cols - 9; x++)
//    {
//        for(int y = 9; y < _image3D.rows - 9; y++)
//        {
//            float weight = sqrt(img_weight.at<float>(y,x));
//            float x_grad = img_x_grad.at<float>(y,x);
//            float y_grad = img_y_grad.at<float>(y,x);
//            //float weight = img_weight.at<float>(y,x);
//            //cout << x_grad << " " << y_grad << endl;

//            img_J0.at<float>((y) , (x)) = x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 0) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 0);
//            img_J1.at<float>((y) , (x)) = x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 1) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 1);
//            img_J2.at<float>((y) , (x)) = x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 2) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 2);
//            img_J3.at<float>((y) , (x)) = x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 3) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 3);
//            img_J4.at<float>((y) , (x)) = x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 4) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 4);
//            img_J5.at<float>((y) , (x)) = x_grad*img_Jw.at<float>((y)*2 + 0, (x)*6 + 5) + y_grad*img_Jw.at<float>((y)*2 + 1, (x)*6 + 5);

//        }
//    }
//    cv::Mat img_J0_show;
//    cv::normalize(img_J0, img_J0_show, 0, 255, cv::NORM_MINMAX);
//    img_J0_show.convertTo(img_J0_show, CV_8U);
//    cv::imshow("J0", img_J0_show);

//    cv::Mat img_J1_show;
//    cv::normalize(img_J1, img_J1_show, 0, 255, cv::NORM_MINMAX);
//    img_J1_show.convertTo(img_J1_show, CV_8U);
//    cv::imshow("J1", img_J1_show);

//    cv::Mat img_J2_show;
//    cv::normalize(img_J2, img_J2_show, 0, 255, cv::NORM_MINMAX);
//    img_J2_show.convertTo(img_J2_show, CV_8U);
//    cv::imshow("J2", img_J2_show);

//    cv::Mat img_J3_show;
//    cv::normalize(img_J3, img_J3_show, 0, 255, cv::NORM_MINMAX);
//    img_J3_show.convertTo(img_J3_show, CV_8U);
//    cv::imshow("J3", img_J3_show);

//    cv::Mat img_J4_show;
//    cv::normalize(img_J4, img_J4_show, 0, 255, cv::NORM_MINMAX);
//    img_J4_show.convertTo(img_J4_show, CV_8U);
//    cv::imshow("J4", img_J4_show);

//    cv::Mat img_J5_show;
//    cv::normalize(img_J5, img_J5_show, 0, 255, cv::NORM_MINMAX);
//    img_J5_show.convertTo(img_J5_show, CV_8U);
//    cv::imshow("J5", img_J5_show);

    // ------------------------------------
    // -----Compute hessian matrix-----
    // ------------------------------------
    cv::Mat hessian(6, 6, CV_32F, cv::Scalar(0));
    for(int x = edge1+edge2; x < _image3D.cols - edge1; x++)
    {
        for(int y = edge1; y < _image3D.rows - edge1; y++)
        {
            float weight = sqrt(img_weight.at<float>(y,x));

            for(int i = 0; i < 6; i++)
            {
                for(int j = 0; j < 6; j++)
                {
                    hessian.at<float>(i,j) += img_J.at<float>((y), (x)*6 + i)*img_J.at<float>((y), (x)*6 + j)*weight;
                }
            }
        }
    }
    _hessian = hessian.clone();

    // ------------------------------------
    // -----Compute sd para matrix-----
    // ------------------------------------

    cv::Mat sdpara_mat(6, 1, CV_32F, cv::Scalar(0));
    for(int x = edge1+edge2; x < _image3D.cols - edge1; x++)
    {
        for(int y = edge1; y < _image3D.rows - edge1; y++)
        {                 float weight = img_weight.at<float>(y,x);
            float residual = img_residual.at<float>(y,x);
            //float temp_img_J = img_J.at<float>((y) , (x)*6 + 0);
            sdpara_mat.at<float>(0) += (weight)*img_J.at<float>((y) , (x)*6 + 0)*residual;
            sdpara_mat.at<float>(1) += (weight)*img_J.at<float>((y) , (x)*6 + 1)*residual;
            sdpara_mat.at<float>(2) += (weight)*img_J.at<float>((y) , (x)*6 + 2)*residual;
            sdpara_mat.at<float>(3) += (weight)*img_J.at<float>((y) , (x)*6 + 3)*residual;
            sdpara_mat.at<float>(4) += (weight)*img_J.at<float>((y) , (x)*6 + 4)*residual;
            sdpara_mat.at<float>(5) += (weight)*img_J.at<float>((y) , (x)*6 + 5)*residual;
        }
    }
    //cout << "sdpara is " << sdpara_mat << endl;
    double num = cols*rows;
    sdpara_mat.convertTo(sdpara_mat, CV_32F, -1.0);
    _sdpara = sdpara_mat.clone();


    //cv::waitKey(0);
}

