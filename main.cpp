#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int ac, char** av)
{
  std::cout<<"q"<<std::endl;

  Mat img1;
  Mat img2;

  img1= imread("../data/test1.png", IMREAD_COLOR );
  if ( !img1.data )
  {
    printf("No image data \n");
    return -1;
  }

  img2= imread("../data/test1.png", IMREAD_COLOR );
  if ( !img2.data )
  {
    printf("No image data \n");
    return -1;
  }

  resize(img1, img1, cv::Size(img1.cols * 0.5,img1.rows * 0.5), INTER_LINEAR);
  resize(img2, img2, cv::Size(img2.cols * 0.5,img2.rows * 0.5), INTER_LINEAR);

  namedWindow("Display Image1 ", WINDOW_AUTOSIZE );
  imshow("Display Image1", img1);
  namedWindow("Display Image2", WINDOW_AUTOSIZE );
  imshow("Display Image2", img2);


  //-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
  int minHessian = 400;
  Ptr<SIFT> detector = SIFT::create( minHessian );
  std::vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;
  detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
  detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );

  std::cout<<"keypoints1 size = "<<keypoints1.size()<<std::endl;
  std::cout<<"keypoints2 size = "<<keypoints2.size()<<std::endl;

  //-- Step 2: Matching descriptor vectors with a FLANN based matcher
  // Since SURF is a floating-point descriptor NORM_L2 is used
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  std::vector< std::vector<DMatch> > knn_matches;
  matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );


  //-- Filter matches using the Lowe's ratio test
  const float ratio_thresh = 0.7f;
  std::vector<DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++)
  {
    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    {
      good_matches.push_back(knn_matches[i][0]);
    }
  }

  //-- Draw matches
  Mat img_matches;
  drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
      Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  //-- Show detected matches
  imshow("Good Matches", img_matches );

  waitKey(0);
  return 0;
}
