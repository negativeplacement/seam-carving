#include <c++/15.2.0/algorithm>
#include <c++/15.2.0/cstdlib>
#include <c++/15.2.0/iostream>
#include <c++/15.2.0/map>
#include <c++/15.2.0/set>
#include <c++/15.2.0/string>
#include <c++/15.2.0/vector>
#include <omp.h>
#include <opencv4/opencv2/opencv.hpp>

struct Object{
    cv::Mat image, mask;
    cv::Point position;
    bool highlight;
};

cv::Mat seam_insertion(cv::Mat& image, std::vector<std::vector<int32_t>>& seam_batch);
cv::Mat seam_removal(cv::Mat& image, std::vector<int32_t>& seam);
std::vector<int32_t> backward_energy_accumulation(cv::Mat& image, cv::Mat *cumulative_energy_map = nullptr, cv::Mat *image_mask = nullptr, double penalty = 0);
std::vector<int32_t> forward_energy_accumulation(cv::Mat& image, cv::Mat *cumulative_energy_map = nullptr, cv::Mat *image_mask = nullptr, double penalty = 0);
void aspect_ratio_change(cv::Mat& image, int32_t width, int32_t height);
void optimal_seam_order(cv::Mat& image, int32_t width, int32_t height);
void content_amplififcation(cv::Mat& image, double factor);

void scale(cv::Mat &image, int32_t dimension, bool _);


void onMouse(int event, int x, int y, int flags, void *data);
void object_removal(cv::Mat& image, cv::Mat& image_mask);

int main(int argc, char* argv[]){
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if(image.empty()){
        std::cerr << "Error Loading Image "<< std::endl;
        return 0;
    }
    
    // object selection
    /*Object object;
    object.image = image.clone();
    object.mask = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallBack("Object Selection", onMouse, &object);
    while(1){
        cv::imshow("Image", object.image);
        if (cv::waitKey(1) == 27) break;
    }
    cv::Mat image_object;
    image.copyTo(image_object, object.mask);
    */

    /*char *transient = NULL;
    int32_t width = std::strtol(argv[2], &transient, 10);
    if(*transient != '\0'){
        std::cerr << "Invalid Argument" <<std::endl;
        return 0;
    }
    int32_t height = std::strtol(argv[3], &transient, 10);
    if(*transient != '\0'){
        std::cerr << "Invalid Argument" <<std::endl;
        return 0;
    }*/

    cv::Mat clone = image.clone();
    //scale(clone, 700, 0);
    //scale(clone, 324, 0);
    //optimal_seam_order(clone, 540, 540);
    cv::imwrite("scale.jpg", clone);
    return 1;
}

cv::Mat seam_insertion(cv::Mat& image, std::vector<std::vector<int32_t>>& seam_batch){
    cv::Mat image_(image.rows, image.cols + seam_batch.size(), image.type());
    #pragma omp parallel for
    for(int32_t _ = 0; _ < image.rows; ++_){
        std::vector<int32_t> seam;
        seam.reserve(seam_batch.size());
        for(std::vector<int32_t>& seam_ : seam_batch) seam.push_back(seam_[_]);
        std::sort(seam.begin(), seam.end());
        uchar *source = image.ptr<uchar>(_), *destination = image_.ptr<uchar>(_);
        int32_t source_ = 0, destination_ = 0;
        for(int32_t index : seam){
            std::memcpy(destination + (destination_ * image.elemSize()), source + (source_ * image.elemSize()), (index - source_ + 1) * image.elemSize());
            destination_ += index - source_ + 1;
            source_ += index - source_ + 1;
            uchar *left = source + (index * image.elemSize()), *right = (index < image.cols - 1) ? left + image.elemSize() : left;
            cv::Vec3b average_pixel;
            for(int8_t channel = 0; channel < image.elemSize(); ++channel) average_pixel[channel] = (static_cast<int>(left[channel]) + static_cast<int>(right[channel]))/2;
            std::memcpy(destination + (destination_ * image.elemSize()), &average_pixel, image.elemSize());
            destination_++;
        }
        if(source_ < image.cols) std::memcpy(destination + (destination_  * image.elemSize()), source + (source_ * image.elemSize()), (image.cols - source_) * image.elemSize());
    }
    return image_;
}

cv::Mat seam_removal(cv::Mat& image, std::vector<int32_t>& seam){
    cv::Mat image_(image.rows, image.cols - 1, image.type());
    #pragma omp parallel for
    for(int32_t _ = 0; _ < image.rows; ++_){
        uchar *source = image.ptr<uchar>(_), *destination = image_.ptr<uchar>(_);
        std::memcpy(destination, source, seam[_] * image.elemSize());
        std::memcpy(destination + (seam[_] * image.elemSize()), source + ((seam[_] * image.elemSize()) + image.elemSize()), (image.cols - seam[_] - 1) * image.elemSize());
    }
    return image_;
}

std::vector<int32_t> backward_energy_accumulation(cv::Mat& image, cv::Mat *cumulative_energy_map, cv::Mat *image_mask, double penalty){
    cv::Mat greyscale_image, greyscale_blur, horizontal_gradient, vertical_gradient, kernel_filter, energy_map;
    cv::cvtColor(image, greyscale_image, cv::COLOR_BGR2GRAY);
    /*cv::Mat median_blur;
    cv::medianBlur(greyscale_image, median_blur, 3);
    cv::Mat salt_mask = greyscale_image == 255, pepper_mask = greyscale_image == 0, filter_mask, absolute_difference, delta_filter;
    cv::bitwise_or(salt_mask, pepper_mask, filter_mask);
    cv::absdiff(greyscale_image, median_blur, absolute_difference);
    cv::threshold(absolut_difference, delta_filter, 20, 255, THRESH_BINARY);
    cv::Mat noise_mask;
    cv::bitwise(delta_filter, filter_mask, noise_mask);
    int32_t noise_pixel = cv::countNonZero(noise_mask);
    double percentage = (noise_pixel / (image.rows * image.cols)) * 100.0;
    if(percentage >= 0.1) greyscale_blur = median_blur;
    
    cv::Mat kernel = (cv::Mat_<int>(3, 3)<< 1, -2, 1, -2, 4, -2, 1, -2, 1);
    cv::filter2D(greyscale_image, kernel_filter, CV_64F, kernel);
    double absolute = cv::sum(cv::abs(kernel_filter))[0];
    double sigma = (absolute * std::sqrt(0.5 * CV_PI)) / (6.0 * (image.rows - 2) * (image.cols - 2));
    if(sigma > 1.0 && sigma < 5.0) cv::bilateralFilter(greyscale_image, greyscale_blur, -1, 2 * sigma, 25.0);
    else if(sigma > 5.0) cv::GaussianBlur(greyscale_image, greyscale_blur, cv::Size(5, 5), 0);*/
    cv::GaussianBlur(greyscale_image, greyscale_blur, cv::Size(5, 5), 0);
    //greyscale_blur = greyscale_image;
    cv::Sobel(greyscale_blur, horizontal_gradient, CV_64F, 1, 0, 3);
    cv::Sobel(greyscale_blur, vertical_gradient, CV_64F, 0, 1, 3);
    cv::magnitude(horizontal_gradient, vertical_gradient, energy_map);
    if(image_mask != nullptr) energy_map.setTo(penalty, *image_mask);
    for(int32_t x = 1; x < energy_map.rows; ++x){
        double *previous_energy = energy_map.ptr<double>(x - 1), *current_energy = energy_map.ptr<double>(x);
        for(int32_t y = 0; y < energy_map.cols; ++y){
            double left = y > 0 ? previous_energy[y - 1] : std::numeric_limits<double>::infinity();
            double center = previous_energy[y];
            double right = (y < energy_map.cols - 1) ? previous_energy[y + 1] : std::numeric_limits<double>::infinity();
            current_energy[y] += std::min({left, center, right});
        }
    }
    if(cumulative_energy_map != nullptr) energy_map.copyTo(*cumulative_energy_map);
    int32_t arbitrary = 0;
    double minimum_energy = std::numeric_limits<double>::infinity(), *energy = energy_map.ptr<double>(energy_map.rows - 1);
    for(int32_t _ = 0; _ < energy_map.cols; ++_){
        if(energy[_] < minimum_energy){
            arbitrary = _;
            minimum_energy = energy[_];
        }
    }
    std::vector<int32_t> seam(energy_map.rows);
    seam[energy_map.rows - 1] = arbitrary;
    for(int32_t _ = energy_map.rows - 1; _ > 0; --_){
        energy = energy_map.ptr<double>(_ - 1);
        int32_t left_index = std::max<int32_t>(0, arbitrary - 1), right_index = std::min<int32_t>(energy_map.cols - 1, arbitrary + 1);
        double left = energy[left_index], center = energy[arbitrary], right = energy[right_index];
        if(left <= center && left <= right) arbitrary = left_index;
        else if(right <= left && right <= center) arbitrary = right_index;
        seam[_ - 1] = arbitrary;
    }
    return seam;
}

std::vector<int32_t> forward_energy_accumulation(cv::Mat& image, cv::Mat *cumulative_energy_map, cv::Mat *image_mask, double penalty){
    cv::Mat greyscale_image;
    cv::cvtColor(image, greyscale_image, cv::COLOR_BGR2GRAY);
    cv::Mat energy_map = cv::Mat::zeros(image.rows, image.cols, CV_64F);
    for(int32_t x = 1; x < image.rows; ++x){
        uchar *previous_intensity = greyscale_image.ptr<uchar>(x - 1), *current_intensity = greyscale_image.ptr<uchar>(x);
        double *previous_energy = energy_map.ptr<double>(x - 1), *current_energy = energy_map.ptr<double>(x);
        uchar *mask = nullptr; image_mask != nullptr && (mask = image_mask->ptr<uchar>(x), 1);
        for(int32_t y = 0; y < image.cols; ++y){
            int32_t left_index = std::max<int32_t>(0, y - 1), right_index = std::min<int32_t>(image.cols - 1, y + 1);
            double center_cost = std::abs(current_intensity[right_index] - current_intensity[left_index]);
            double left_cost = center_cost + std::abs(previous_intensity[y] - current_intensity[left_index]);
            double right_cost = center_cost + std::abs(previous_intensity[y] - current_intensity[right_index]);
            double center = previous_energy[y] + center_cost;
            double left = y > 0 ? previous_energy[left_index] + left_cost : std::numeric_limits<double>::infinity();
            double right = (y < image.cols - 1) ? previous_energy[right_index] + right_cost : std::numeric_limits<double>::infinity();
            current_energy[y] += std::min({left, center, right});
            (image_mask != nullptr && mask[y]) && (current_energy[y] += penalty, 1);
        }
    }
    if(cumulative_energy_map != nullptr) energy_map.copyTo(*cumulative_energy_map);
    int32_t arbitrary = 0;
    double minimum_value = std::numeric_limits<double>::infinity(), *energy = energy_map.ptr<double>(image.rows - 1);
    for(int32_t _ = 0; _ < image.cols; ++_){
        if(energy[_] < minimum_value){
            arbitrary = _;
            minimum_value = energy[_];
        }
    }
    std::vector<int32_t> seam(image.rows);
    seam[image.rows - 1] = arbitrary;
    for(int32_t _ = image.rows - 1; _ > 0; --_){
        energy = energy_map.ptr<double>(_ - 1);
        double energy_ = energy_map.at<double>(_, arbitrary);
        uchar *previous_intensity = greyscale_image.ptr<uchar>(_ - 1), *current_intensity = greyscale_image.ptr<uchar>(_);
        int32_t left_index = std::max<int32_t>(0, arbitrary - 1), right_index = std::min<int32_t>(image.cols - 1, arbitrary + 1);
        double center_cost = std::abs(current_intensity[right_index] - current_intensity[left_index]);
        double left_cost = center_cost + std::abs(previous_intensity[arbitrary] - current_intensity[left_index]);
        double right_cost = center_cost + std::abs(previous_intensity[arbitrary] - current_intensity[right_index]);
        if(std::abs(energy_ - (energy[left_index] + left_cost)) <1e-6) arbitrary = left_index;
        else if(std::abs(energy_ - (energy[right_index] + right_cost)) <1e-6) arbitrary = right_index;
        seam[_ - 1] = arbitrary;
    }
    return seam;
}

void scale(cv::Mat &image, int32_t dimension, bool _){
    if(_) cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
    if(image.cols < dimension){
        std::vector<std::vector<int32_t>> seam_batch;
        cv::Mat image_ = image.clone();
        for(int i = 0; i < dimension - image.cols; ++i){
            std::vector<int32_t> seam = backward_energy_accumulation(image_);
            seam_batch.push_back(seam);
            image_ = seam_removal(image_, seam);
        }
        image = seam_insertion(image, seam_batch);
    } else{
        while(image.cols > dimension){
            std::vector<int32_t> seam = backward_energy_accumulation(image);
            image = seam_removal(image, seam);
        }        
    }
    if(_) cv::rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE);
    return;
}

void aspect_ratio_change(cv::Mat& image, int32_t width, int32_t height){
    if(image.cols < width){
        std::vector<std::vector<int32_t>> seam_batch;
        cv::Mat image_ = image.clone();
        for(int32_t _ = 0; _ < width - image.cols; ++_){
            std::vector<int32_t> seam = forward_energy_accumulation(image_);
            seam_batch.push_back(seam);
            image_ = seam_removal(image_, seam);
        }
        image = seam_insertion(image, seam_batch);
    } else{
        while(image.cols > width){
            std::vector<int32_t> seam = forward_energy_accumulation(image);
            image = seam_removal(image, seam);
        }
    }
    cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
    if(image.cols < height){
        std::vector<std::vector<int32_t>> seam_batch;
        cv::Mat image_ = image.clone();
        for(int32_t _ = 0; _ < height - image.cols; ++_){
            std::vector<int32_t> seam = forward_energy_accumulation(image_);
            seam_batch.push_back(seam);
            image_ = seam_removal(image_, seam);
        }
        image = seam_insertion(image, seam_batch);
    } else{
        while(image.cols > height){
            std::vector<int32_t> seam = forward_energy_accumulation(image);
            image = seam_removal(image, seam);
        }
    }
    cv::rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE);
    return;
}

void optimal_seam_order(cv::Mat& image, int32_t width, int32_t height){
    cv::Mat cumulative_energy_map;
    while(image.cols != width || image.rows != height){
        std::vector<std::vector<int32_t>> horizontal_seam, vertical_seam;
        double horizontal_minimum = 0, vertical_minimum = 0;
        if(image.cols != width){
            if(image.cols < width){
                cv::Mat image_ = image.clone();
                for(int32_t _ = 0; _ < width - image.cols; ++_){
                    std::vector<int32_t> seam = forward_energy_accumulation(image_, &cumulative_energy_map);
                    double minimum_energy = cumulative_energy_map.at<double>(cumulative_energy_map.rows - 1, seam[seam.size() - 1]);
                    horizontal_minimum += minimum_energy / cumulative_energy_map.rows;
                    horizontal_seam.push_back(seam);
                    image_ = seam_removal(image_, seam);
                }
                horizontal_minimum /= (width - image.cols);
            } else{
                std::vector<int32_t> seam = forward_energy_accumulation(image, &cumulative_energy_map);
                double minimum_energy = cumulative_energy_map.at<double>(cumulative_energy_map.rows - 1, seam[seam.size() - 1]);
                horizontal_minimum = minimum_energy / cumulative_energy_map.rows;
                horizontal_seam.push_back(seam);
            }
        }
        if(image.rows != height){
            cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
            if(image.cols < height){
                cv::Mat image_ = image.clone();
                for(int32_t _ = 0; _ < height - image.cols; ++_){
                    std::vector<int32_t> seam = forward_energy_accumulation(image_, &cumulative_energy_map);
                    double minimum_energy = cumulative_energy_map.at<double>(cumulative_energy_map.rows - 1, seam[seam.size() - 1]);
                    vertical_minimum += minimum_energy / cumulative_energy_map.rows;
                    vertical_seam.push_back(seam);
                    image_ = seam_removal(image_, seam);
                }
                vertical_minimum /= (height - image.cols);
            } else{
                std::vector<int32_t> seam = forward_energy_accumulation(image, &cumulative_energy_map);
                double minimum_energy = cumulative_energy_map.at<double>(cumulative_energy_map.rows - 1, seam[seam.size() - 1]);
                vertical_minimum = minimum_energy / cumulative_energy_map.rows;
                vertical_seam.push_back(seam);
            }
            cv::rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE);
        }
        std::cout << "Horizontal: " << horizontal_minimum << std::endl;
        std::cout << "Vertical: " << vertical_minimum << "\n" << std::endl;
        if(horizontal_minimum < vertical_minimum && image.cols != width){
            if(!horizontal_seam.empty()) image = image.cols <= width ? seam_insertion(image, horizontal_seam) : seam_removal(image, horizontal_seam[0]);
        } else /*if(vertical_minimum < horizontal_minimum && image.rows != height)*/{
            cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
            if(!vertical_seam.empty()) image = image.cols < height ? seam_insertion(image, vertical_seam) : seam_removal(image, vertical_seam[0]);
            cv::rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE);
        }
    }
    return;
}

void content_amplififcation(cv::Mat& image, double factor){
    int32_t width = image.cols, height = image.rows;
    cv::resize(image, image, cv::Size(0,0), factor, factor, cv::INTER_CUBIC);
    optimal_seam_order(image, width, height);
    return;
}

void onMouse(int event, int x, int y, int flags, void *data){
    Object *object = (Object*)data;
    bool highlight = 0;
    cv::Point position(x, y);
    if(event == cv::EVENT_LBUTTONDOWN){
        highlight = 1;
        object->position = position;
        cv::circle(object->mask, position, 10, cv::Scalar(255), -1);
        cv::circle(object->image, position, 10, cv::Scalar(0, 255, 0), -1);
    }  else if(highlight && event == cv::EVENT_MOUSEMOVE){
        cv::line(object->mask, object->position, position, cv::Scalar(255), 10);
        cv::line(object->image, object->position, position, cv::Scalar(0, 255, 0), 10);
        position = position;
    } else if(event == cv::EVENT_LBUTTONUP){
        highlight = 0;
    }
    return;
}

void object_removal(cv::Mat& image, cv::Mat& image_mask){
    int32_t width = image.cols;
    cv::Mat greyscale_image, greyscale_blur, horizontal_gradient, vertical_gradient, energy_map;
    while(cv::countNonZero(image_mask) > 0){
        std::vector<int32_t> seam = forward_energy_accumulation(image, nullptr, &image_mask, 1e-9);
        image = seam_removal(image, seam);
        image_mask = seam_removal(image_mask, seam);
    }
    if(width - image.cols > 0){
        std::vector<std::vector<int32_t>> seam_batch;
        cv::Mat image_ = image.clone();
        for(int32_t _ = 0; _ < width - image.cols; ++_){
            std::vector<int32_t> seam = forward_energy_accumulation(image_);
            seam_batch.push_back(seam);
            image_ = seam_removal(image_, seam);
        }
        image = seam_insertion(image, seam_batch);
    }
    return;
}
// c_cpp -file avidan_shamir_seam_carving.cpp -argument sample.jpg -link opencv -flag "-fopenmp","-O2","-Wa,-mbig-obj"