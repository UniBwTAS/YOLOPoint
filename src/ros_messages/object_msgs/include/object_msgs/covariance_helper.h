#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>

#include <object_msgs/Object.h>

namespace object_msgs
{

enum RandomVariables
{
    X = 0,
    Y,
    Z,
    RX,
    RY,
    RZ,
    dX,
    dY,
    dZ,
    dRX,
    dRY,
    dRZ,
    ddX,
    ddY,
    ddZ,
    ddRX,
    ddRY,
    ddRZ,
    LF,
    LR,
    WL,
    WR,
    HT,
    HB
};

class CovarianceHelper
{
  public:
    static Eigen::MatrixXf covarianceFromMsg(const object_msgs::Object& msg)
    {
        // inverse of gauss sum formula
        int matrix_size = static_cast<int>(std::round(std::sqrt(2. * msg.complete_covariance.size() + 0.25) - 0.5));
        Eigen::MatrixXf cov(matrix_size, matrix_size);
        int i = 0;
        for (int row = 0; row < matrix_size; ++row)
        {
            for (int col = row; col < matrix_size; ++col)
            {
                cov(row, col) = msg.complete_covariance[i];
                cov(col, row) = msg.complete_covariance[i++];
            }
        }
        return cov;
    }

    static void covarianceToMsg(Eigen::MatrixXf& cov, object_msgs::Object& msg, const boost::array<uint8_t, 24> mask)
    {
        int matrix_size = cov.rows();
        msg.complete_covariance.clear();
        for (int row = 0; row < matrix_size; ++row)
        {
            for (int col = row; col < matrix_size; ++col)
            {
                msg.complete_covariance.push_back(cov(row, col));
            }
        }
        msg.state_validity = mask;
    }

    static Eigen::MatrixXf getSubMatrix(Eigen::MatrixXf& cov, const std::vector<int>& indices)
    {
        Eigen::MatrixXf sub(indices.size(), indices.size());
        for (int row = 0; row < indices.size(); ++row)
        {
            for (int col = row; col < indices.size(); ++col)
            {
                sub(row, col) = cov(indices[row], indices[col]);
                sub(col, row) = cov(indices[col], indices[row]);
            }
        }
        return sub;
    }

    static void setSubMatrix(Eigen::MatrixXf& cov, const std::vector<int>& indices, const Eigen::MatrixXf& sub)
    {
        for (int row = 0; row < indices.size(); ++row)
        {
            for (int col = row; col < indices.size(); ++col)
            {
                cov(indices[row], indices[col]) = sub(row, col);
                cov(indices[col], indices[row]) = sub(col, row);
            }
        }
    }

    static Eigen::MatrixXf reshapeByRandomVariables(const Eigen::MatrixXf& in_cov,
                                                    const std::vector<RandomVariables>& in_rvs,
                                                    const std::vector<RandomVariables>& out_rvs,
                                                    bool rvs_sorted = true)
    {
        Eigen::MatrixXf out_cov(out_rvs.size(), out_rvs.size());
        std::vector<int> map_out_idxs_to_in_idxs(out_rvs.size());
        int next_search_start_idx = 0;
        for (int i = 0; i < out_rvs.size(); ++i)
        {
            bool rv_found = false;
            for (int j = next_search_start_idx; j < in_rvs.size(); ++j)
            {
                if (out_rvs[i] == in_rvs[j])
                {
                    map_out_idxs_to_in_idxs[i] = j;
                    if (rvs_sorted)
                        next_search_start_idx = j + 1;
                    rv_found = true;
                    break;
                }
            }
            if (!rv_found)
            {
                map_out_idxs_to_in_idxs[i] = -1;
            }
        }
        for (int row = 0; row < out_rvs.size(); ++row)
        {
            for (int col = row; col < out_rvs.size(); ++col)
            {
                int in_idx_row = map_out_idxs_to_in_idxs[row];
                int in_idx_col = map_out_idxs_to_in_idxs[col];
                if (in_idx_row >= 0 && in_idx_col >= 0)
                {
                    out_cov(row, col) = in_cov(in_idx_row, in_idx_col);
                    out_cov(col, row) = in_cov(in_idx_row, in_idx_col);
                }
                else
                {
                    out_cov(row, col) = 0;
                    out_cov(col, row) = 0;
                }
            }
        }
        return out_cov;
    }

    static boost::array<uint8_t, 24> getMaskFromRandomVariables(const std::vector<RandomVariables>& rvs)
    {
        boost::array<uint8_t, 24> a{};
        for (RandomVariables rv : rvs)
        {
            a[rv] = 1;
        }
        return a;
    }

    static std::vector<RandomVariables> getRandomVariablesFromMask(const boost::array<uint8_t, 24>& mask)
    {
        std::vector<RandomVariables> rvs;
        for (int rv = 0; rv < 24; ++rv)
        {
            if (mask[rv])
            {
                rvs.push_back(static_cast<RandomVariables>(rv));
            }
        }
        return rvs;
    }

    static boost::array<uint8_t, 24> invertMask(boost::array<uint8_t, 24> mask)
    {

        for (int rv = 0; rv < 24; ++rv)
        {
            mask[rv] = !mask[rv];
        }

        return mask;
    }

    static bool isCovarianceValid(const object_msgs::Object& msg, long rv_dim=-1)
    {
        // get dimension of random variable
        rv_dim = rv_dim < 0 ? std::count(msg.state_validity.begin(), msg.state_validity.end(), true) : rv_dim;
        // get expected elements for a upper triangle of a matrix (by gaussian sum formula)
        long upper_triangle_count = (rv_dim * (rv_dim + 1)) / 2;
        // check whether expected number of elements
        return upper_triangle_count == msg.complete_covariance.size();
    }
};
} // namespace object_msgs
