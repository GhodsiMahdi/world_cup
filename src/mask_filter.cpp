/*
written by Mahdi
fitst a mask to a detected face using opencv and dlib
*/
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <dlib/image_io.h>
#include <algorithm>


using namespace std;
using namespace dlib;

cv::Mat_<cv::Vec4b> createAlphaImage(const cv::Mat& mat,
                                      cv::Mat &alpha);

bool addGaussianNoise(cv::Mat &mSrc,
                              cv::Mat &mDst,
                              double Mean,
                              double StdDev);

cv::Mat equalizeIntensity(const cv::Mat& inputImage);

void maskMatch(int mask_group, dlib::rectangle mrect,
                dlib::full_object_detection shape,
                std::vector<dlib::point> &from,
                std::vector<dlib::point> &to,
                std::vector<dlib::point> &outline);

cv::Mat_<cv::Vec4b> MatchColor(cv::Mat &source,
                                cv::Mat &target);

int main(int argc, char** argv) try
{
    if (argc < 2)
    {
        cout << "Call this program like this:" << endl;
        cout << "./mask_filter_ex  your_image_folder/*.jpg" << endl;
        cout << "You need to place shape_predictor_68_face_landmarks.dat ";
        cout << "file where your exectutable file is. You can download from:\n";
        cout << "http://dlib.net/files/" << endl;
        return 0;
    }

    cv::Mat img_out, mask_img_raw, input_bgr, input_rgb, mask_img, test_img;
    dlib::frontal_face_detector fdetector;
    dlib::shape_predictor sppredictor;
    fdetector = dlib::get_frontal_face_detector();
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sppredictor;
    dlib::image_window win;

    for (int i = 1; i < argc; ++i)
    {
        input_bgr = cv::imread(argv[i]);
        //opencv reads default bgr but dlib works with rgb, read up aboout
        //color space systems if you need to, there are tons of them
        cv::cvtColor(input_bgr, input_rgb, CV_BGR2RGB);

        //read one mask at random from the folder containing group of masks
        int number_of_masks_in_group = 87;
        srand(time(0));
        int mask_number = std::rand() % number_of_masks_in_group + 1;
        mask_img_raw = cv::imread(
          "/home/aupera/Mahdi/Projects/Data/masks_grouped/1_old/"
           + std::to_string(mask_number) + ".png", cv::IMREAD_UNCHANGED);

        //mask_img = MatchColor(input_bgr, mask_img_raw);
        mask_img = mask_img_raw;
        dlib::matrix<dlib::rgb_pixel> img, raw_input_img;
        dlib::matrix<dlib::rgb_alpha_pixel> mask;
        dlib::cv_image<dlib::rgb_pixel> cvimg(input_rgb);
        dlib::cv_image<dlib::rgb_alpha_pixel> maskImg(mask_img);
        dlib::assign_image(mask, maskImg);
        dlib::assign_image(img, cvimg);
        dlib::assign_image(raw_input_img, img);

        dlib::pyramid_up(img);

        std::vector<rectangle> dets = fdetector(img);
        std::cout << "Number of faces detected: " << dets.size() << std::endl;
        // get the landdmkars and put them in shape
        std::vector<dlib::full_object_detection> shapes;
        cv::Mat output, src_mask;
        for (unsigned long j = 0; j < dets.size(); ++j)
        {
            dlib::full_object_detection shape = sppredictor(img, dets[j]);
            //just to illustrate how it is and how to access landmark positions
            std::cout << "number of parts: "<< shape.num_parts() << std::endl;
            std::cout << "pixel position of first part (left ear):  "
            << shape.part(0) << std::endl;
            std::cout << "pixel position of second part: " << shape.part(1)
             << std::endl;
            //you cna delete the above logging code aboout pixel pos and numbers
            const dlib::rgb_pixel color(0,255,0);
            std::vector<dlib::point> from;
            std::vector<dlib::point> to;
            std::vector<dlib::point> outline;
            // Draw the mask onto the image
            auto mrect = dlib::get_rect(mask);
            //matching points placed in from and to look at function for details
            //the mask group number is hard coded here but change accordingly
            int mask_group_number = 1;
            maskMatch(mask_group_number, mrect, shape, from, to, outline);
            //literally draw the scaled mask after calling dlib similarity func
            auto tform = find_similarity_transform(from, to);
            for (long r = 0; r < mask.nr(); ++r)
            {
                for (long c = 0; c < mask.nc(); ++c)
                {
                    dlib::point p = tform(dlib::point(c,r));
                    if (dlib::get_rect(img).contains(p))
                        dlib::assign_pixel(img(p.y(),p.x()), mask(r,c));
                }
            }




            cv::Mat img1 = dlib::toMat(img);
            cv::Mat img2 = dlib::toMat(raw_input_img);
            cv::cvtColor(img1, img1, CV_RGB2BGR);
            cv::cvtColor(img2, img2, CV_RGB2BGR);
            //src_mask = cv::Mat::zeros(outline[5].y() - outline[0].y(), outline[2].x() - outline[0].x(), mask_img_raw.depth());
            src_mask = cv::Mat::zeros(img1.rows, img1.cols, img1.depth());
            //cv::Mat maskii = cv::Mat::ones(img1.rows, img1.cols, mask_img_raw.depth());
            std::cout << "image wxh is " << src_mask.cols << "x" << src_mask.rows << std::endl;
            std::vector<cv::Point> points(6);
            cv::Point poly[1][6];
            for (int outline_index = 0; outline_index < outline.size(); outline_index++)
            {
              auto x = outline[outline_index].x();// - outline[0].x();
              auto y = outline[outline_index].y();// - outline[0].y();
              poly[0][outline_index] = cv::Point(std::max(int(x),0), std::max(int(y),0));
              points[outline_index] =  cv::Point(std::max(int(x),0), std::max(int(y),0));
              std::cout << "point " << outline_index << " is " << points[outline_index] << std::endl;
            }




            const cv::Point* polygons[1] = { poly[0] };
            int num_points[] = { 6 };
            cv::fillPoly(src_mask, polygons, num_points, 1, cv::Scalar(255,255,255));
            dlib::point dlib_center((outline[1] + outline[4]) / 2);
            cv::Point cv_center(dlib_center.x(), dlib_center.y());



            //cv::convexHull(poly[0], hullIndex, false, false);

            //output = img1.clone();
            //cv::resize(mask_img_raw, mask_resized, src_mask.size());
            cv::Mat dest;
            cv::resize(input_bgr, dest, img1.size());

            cv::resize(img2, img2, img1.size(), cv::INTER_AREA);
            cv::GaussianBlur(img2, img2, cv::Size(5,5), 0);
            cv::GaussianBlur(img1, img1, cv::Size(5,5), 0);
            cv::absdiff(img1, img2, test_img);
            cv::cvtColor(test_img, test_img, CV_BGR2GRAY);
            cv::Mat img_th;
            cv::threshold(test_img, test_img, 10, 255, cv::THRESH_BINARY);

            int largest_area=0;
            int largest_contour_index=0;
            cv::Rect bounding_rect;
            std::vector<std::vector<cv::Point>> contours; // Vector for storing contour
            std::vector<cv::Vec4i> hierarchy1;
            cv::findContours( test_img, contours, hierarchy1,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
            // iterate through each contour.
            for( int i = 0; i< contours.size(); i++ )
            {
                //  Find the area of contour
                double a=contourArea( contours[i],false);
                if(a>largest_area){
                    largest_area=a;cout<<i<<" area  "<<a<<endl;
                    // Store the index of largest contour
                    largest_contour_index=i;
                    // Find the bounding rectangle for biggest contour
                    //bounding_rect=boundingRect(contours[i]);
                }
            }
            std::vector<cv::Moments> mu(contours.size() );
            for( int i = 0; i < contours.size(); i++ )
            { mu[i] = cv::moments( contours[i], false ); }

                      ///  Get the mass centers:
            std::vector<cv::Point2f> mc( contours.size() );
            for( int i = 0; i < contours.size(); i++ )
            { mc[i] = cv::Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }


            img_out = cv::Mat::zeros(test_img.rows, test_img.cols, CV_8UC3);
            cv::drawContours( img_out, contours,largest_contour_index, cv::Scalar(255,255,255), cv::FILLED,8,hierarchy1);

            //cv::absdiff(img1, dest, maskii);
            //cv::cvtColor( src_mask, maskii, CV_BGR2GRAY );
            //maskii.convertTo(maskii, 0);
            //src_mask.convertTo(src_mask, CV_8U);
            //std::cout << maskii.cols << "x" << maskii.rows << " " << maskii.channels()<<std::endl;
            //std::cout << src_mask.cols << "x" << src_mask.rows << " " << src_mask.channels() << std::endl;
            //test_img.convertTo(test_img, CV_8U);
            //cv::resize(maskii, maskii, src_mask.size());
            //cv::absdiff(maskii,src_mask, src_mask);

            //cv::cvtColor( src_mask, src_mask, CV_BGR2GRAY );
            //cv::threshold(src_mask, src_mask, 100,255,cv::THRESH_BINARY);
            //cv::cvtColor( test_img, test_img, CV_GRAY2BGR);
            cv::GaussianBlur(src_mask, src_mask, cv::Size(5,5), 0);
            cv::threshold(src_mask, src_mask, 0,255,cv::THRESH_BINARY + cv::THRESH_OTSU);
            //cv::GaussianBlur(img_out, img_out, cv::Size(5,5), 0);
            //cv::threshold(img_out, img_out, 0,255,cv::THRESH_BINARY + cv::THRESH_OTSU);
            cv::seamlessClone(img1, dest, img_out, cv_center, output, cv::NORMAL_CLONE);



          }



        //*****************IF YOU WANT TO VISUALIZE*****************
        //uncomment next 2 lines if you want to illustrate each image
        //win.set_image(img);
        //std::cout << "Hit enter to process the next image." << endl;
        //std::cin.get();
        std::string str(argv[i]);
        std::size_t found = str.find_last_of("/");
        std::string savename = str.substr(found+1);
        //*****************IF YOU WANT TO SAVE RESULTS*****************
        //uncomment if you want to save in images folder the results
        if(dets.size() > 0)
        {
          //cv::blur(output, output, cv::Size(4,4));
          std::cout << "we here dude" << std::endl;
          //dlib::save_jpeg(img, "images/" + savename);
          cv::imwrite("images/"+ savename, img_out);
        }

      }

      std::cout << "Done processing all images in your directory" << std::endl;
      return 0;
}
catch(std::exception& e)
{
    std::cout << e.what() << std::endl;
}

cv::Mat_<cv::Vec4b> createAlphaImage(const cv::Mat& mat,
                                      cv::Mat &alpha)
{
  //makes sure that the alpha channel which contains trnasparency information
  //is added back to the image after it has been processed
  cv::Mat_<cv::Vec4b> dst;
  std::vector<cv::Mat> matChannels;
  cv::split(mat, matChannels);
  matChannels.push_back(alpha);
  cv::merge(matChannels, dst);
  return dst;
}

bool addGaussianNoise(cv::Mat &mSrc,
                              cv::Mat &mDst,double Mean=0.0,
                              double StdDev=100.0)
{
  //this function adds guassian noise to mask, when mean= 0 all the power is
  // about standard deviation, the higher the standard deviation the more
  // noise you get
    if(mSrc.empty())
    {
        std::cout<<"[Error]! Input Image Empty!";
        return 0;
    }
    cv::Mat mSrc_16SC = mSrc.clone();
    cv::Mat mGaussian_noise(mSrc.size(), mSrc.type());
    //cv::randn(mGaussian_noise,cv::Scalar::all(Mean), cv::Scalar::all(StdDev));
    cv::randn(mGaussian_noise,Mean, StdDev);
    cv::addWeighted(mSrc_16SC, 1.0, mGaussian_noise, 1.0, 0.0, mDst);

    return true;
}

cv::Mat equalizeIntensity(const cv::Mat& inputImage)
{
    if(inputImage.channels() >= 3)
    {
        cv::Mat ycrcb;
        cv::cvtColor(inputImage,ycrcb,CV_BGRA2BGR);
        cv::cvtColor(ycrcb,ycrcb,CV_BGR2YCrCb);

        std::vector<cv::Mat> channels;
        cv::split(ycrcb,channels);
        cv::equalizeHist(channels[0], channels[0]);

        cv::Mat result;
        cv::merge(channels,ycrcb);

        cv::cvtColor(ycrcb,result,CV_YCrCb2BGR);

        return result;
    }
    return cv::Mat();
}

void maskMatch(int mask_group, dlib::rectangle mrect,
                dlib::full_object_detection shape,
                std::vector<dlib::point> &from,
                std::vector<dlib::point> &to,
                std::vector<dlib::point> &outline)
{
  //my algorithm to aligh a given mask to a face
  auto top  = shape.part(27);
  auto rear = shape.part(5);
  auto reye = shape.part(36);
  auto nose = shape.part(63);
  auto lear = shape.part(11);
  auto leye = shape.part(45);
  auto lend = shape.part(48);
  auto rend = shape.part(54);
  auto TLear = shape.part(1);
  auto TRear = shape.part(15);
  auto TMnose = shape.part(29);
  auto TMmouth = shape.part(57);
  auto TChin = shape.part(8);
  auto lmask = (TLear + shape.part(36)) / 2 ;
  auto Rmask = (TRear + shape.part(45)) / 2;
  auto MTopMask = TMnose;
  auto MBotMask = TMmouth;
  auto TopMiddleRect = (mrect.tl_corner() + mrect.tr_corner()) / 2;
  auto BottMiddleRect = (mrect.bl_corner() + mrect.br_corner()) / 2;
  auto CenterRect = (TopMiddleRect + BottMiddleRect) / 2;
  auto BotRight1 = dlib::point(TRear.x(),TChin.y());
  auto BotLeft1 = dlib::point(TLear.x(),TChin.y());

  //Only spent time matching first group, feel free  to make more cases for
  //matching other groups
  switch(mask_group)
  {
    case 1:
      from = {mrect.tl_corner(), TopMiddleRect , mrect.tr_corner(),
                CenterRect, BottMiddleRect};
      to = {TLear, TMnose, TRear, TMmouth, TChin};
      outline = {TLear, TMnose, TRear, BotRight1, TChin, BotLeft1};
      break;

    default:
      from = {mrect.tl_corner(), TopMiddleRect ,
                mrect.tr_corner(), CenterRect, BottMiddleRect};
      to = {TLear, TMnose, TRear, TMmouth,
                TChin + (shape.part(63) - shape.part(33))};
      outline = {TLear, TMnose, TRear, BotRight1, TChin, BotLeft1};
      break;
  }

}


cv::Mat_<cv::Vec4b> MatchColor(cv::Mat &source,
                                cv::Mat &target)
{
  cv::Mat mask_img, input_img;
  mask_img = target;
  input_img = source;
  std::vector<cv::Mat> temp(4);
  cv::Mat channelA;
  cv::split(mask_img, temp);
  channelA = temp[3].clone();

  cv::blur(mask_img, mask_img, cv::Size(4,4));
  addGaussianNoise(mask_img, mask_img, 0, 5);
  //mask_img = equalizeIntensity(mask_img);
  //cv::Mat gaussian_noise = img.clone();
  //cv::randn(gaussian_noise,128,30);

  cv::cvtColor(mask_img, mask_img, CV_BGR2Lab);
  cv::cvtColor(input_img, input_img, CV_BGR2Lab);

  std::vector<cv::Mat> mask_lab_channels(3), input_lab_channels(3);
  cv::split(mask_img, mask_lab_channels);
  cv::split(input_img, input_lab_channels);

  std::vector<cv::Scalar> mask_mean_channels(3), mask_stdev_channels(3),
                            input_mean_channels(3), input_stddev_channels(3);

  for (int i = 0; i < 3; i++)
  {
    cv::meanStdDev(mask_lab_channels.at(i),
                    mask_mean_channels[i], mask_stdev_channels[i]);
    cv::meanStdDev(input_lab_channels.at(i),
                    input_mean_channels[i], input_stddev_channels[i]);
  }

  for (int i = 0; i < 3; i++)
  {
    cv::subtract(mask_lab_channels.at(i),
                  mask_mean_channels.at(i), mask_lab_channels[i]);
    cv::multiply(mask_lab_channels.at(i),
                  (mask_stdev_channels.at(i) / input_stddev_channels.at(i)),
                  mask_lab_channels[i]);
    cv::add(mask_lab_channels.at(i),
              input_mean_channels.at(i), mask_lab_channels[i]);
    cv::threshold(mask_lab_channels.at(i), mask_lab_channels[i], 255, 255, 2);
  }

  std::vector<cv::Mat> without_alpha;
  std::vector<cv::Mat> with_alpha;
  cv::Mat tempImage;
  cv::merge(mask_lab_channels, tempImage);
  cv::cvtColor(tempImage, target, CV_Lab2BGR);
  return createAlphaImage(target, channelA);
}
