#include "MODEL/DLModel.hpp"

int main(void)
{
  DLModel model;

  model.train(32);

  model.estimate(
    "public/checkpoints/iter_32_conv_w.bin",
    "public/checkpoints/iter_32_conv_b.bin",
    "public/checkpoints/iter_32_exf_w.bin",
    "public/checkpoints/iter_32_exf_b.bin",
    "public/checkpoints/iter_32_dense_w.bin",
    "public/checkpoints/iter_32_dense_b.bin"
  );

  return 0;
}
