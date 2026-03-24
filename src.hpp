#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Move current query to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Stack all keys[0..i] vertically to create K_stacked
    Matrix *k_stacked = matrix_memory_allocator.Allocate("k_stacked");
    gpu_sim.Copy(keys[0], k_stacked, kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(k_stacked);

    for (size_t j = 1; j <= i; ++j) {
      Matrix *temp_key = matrix_memory_allocator.Allocate("temp_key_" + std::to_string(j));
      gpu_sim.Copy(keys[j], temp_key, kInGpuHbm);
      gpu_sim.MoveMatrixToSharedMem(temp_key);

      Matrix *new_k_stacked = matrix_memory_allocator.Allocate("new_k_stacked_" + std::to_string(j));
      gpu_sim.Concat(k_stacked, temp_key, new_k_stacked, 0, kInSharedMemory);

      gpu_sim.ReleaseMatrix(k_stacked);
      gpu_sim.ReleaseMatrix(temp_key);
      k_stacked = new_k_stacked;
    }

    // Stack all values[0..i] vertically to create V_stacked
    Matrix *v_stacked = matrix_memory_allocator.Allocate("v_stacked");
    gpu_sim.Copy(values[0], v_stacked, kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(v_stacked);

    for (size_t j = 1; j <= i; ++j) {
      Matrix *temp_value = matrix_memory_allocator.Allocate("temp_value_" + std::to_string(j));
      gpu_sim.Copy(values[j], temp_value, kInGpuHbm);
      gpu_sim.MoveMatrixToSharedMem(temp_value);

      Matrix *new_v_stacked = matrix_memory_allocator.Allocate("new_v_stacked_" + std::to_string(j));
      gpu_sim.Concat(v_stacked, temp_value, new_v_stacked, 0, kInSharedMemory);

      gpu_sim.ReleaseMatrix(v_stacked);
      gpu_sim.ReleaseMatrix(temp_value);
      v_stacked = new_v_stacked;
    }

    // Transpose K_stacked
    gpu_sim.Transpose(k_stacked, kInSharedMemory);

    // Compute Q * K^T
    Matrix *qk_t = matrix_memory_allocator.Allocate("qk_t");
    gpu_sim.MatMul(current_query, k_stacked, qk_t);

    // Compute Softmax on rows of QK^T
    // For each row, compute exp(row) / sum(exp(row))
    Matrix *exp_qk_t = matrix_memory_allocator.Allocate("exp_qk_t");
    gpu_sim.MatExp(qk_t, exp_qk_t);

    // For each row, compute the sum and divide
    size_t num_rows = current_query->GetRowNum();
    Matrix *softmax_result = matrix_memory_allocator.Allocate("softmax_result");

    for (size_t row = 0; row < num_rows; ++row) {
      // Get the row from exp_qk_t
      Matrix *row_matrix = matrix_memory_allocator.Allocate("row_" + std::to_string(row));
      gpu_sim.GetRow(exp_qk_t, row, row_matrix, kInSharedMemory);

      // Sum the row
      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(row));
      gpu_sim.Sum(row_matrix, row_sum);

      // Divide the row by its sum
      Matrix *normalized_row = matrix_memory_allocator.Allocate("normalized_row_" + std::to_string(row));
      gpu_sim.MatDiv(row_matrix, row_sum, normalized_row);

      // Set the row in softmax_result
      if (row == 0) {
        // First row, initialize softmax_result
        gpu_sim.Copy(normalized_row, softmax_result, kInSharedMemory);
      } else {
        // Concatenate with existing result
        Matrix *new_softmax = matrix_memory_allocator.Allocate("new_softmax_" + std::to_string(row));
        gpu_sim.Concat(softmax_result, normalized_row, new_softmax, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_result);
        softmax_result = new_softmax;
      }

      gpu_sim.ReleaseMatrix(row_matrix);
      gpu_sim.ReleaseMatrix(row_sum);
      gpu_sim.ReleaseMatrix(normalized_row);
    }

    // Compute Softmax(QK^T) * V
    Matrix *attention_result = matrix_memory_allocator.Allocate("attention_result");
    gpu_sim.MatMul(softmax_result, v_stacked, attention_result);

    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(attention_result);

    // Release intermediate matrices
    gpu_sim.ReleaseMatrix(current_query);
    gpu_sim.ReleaseMatrix(k_stacked);
    gpu_sim.ReleaseMatrix(v_stacked);
    gpu_sim.ReleaseMatrix(qk_t);
    gpu_sim.ReleaseMatrix(exp_qk_t);
    gpu_sim.ReleaseMatrix(softmax_result);

    gpu_sim.Run(false, &matrix_memory_allocator);

    // Commit the answer
    rater.CommitAnswer(*attention_result);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu