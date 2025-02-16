#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        // 其实就是让你推倒m , n
        auto A = inputs[0], B = inputs[1];
        auto shapeA = A->getDims();
        auto shapeB = B->getDims();
        int rankA = A->getRank(); // Rank is the Shape of TensorDims
        int rankB = B->getRank();
        // 这里的实现和infiniTensor有一点不一样，可以不必存储b
        // 推导batch
        // int b = 0;
        Shape shapeA1(shapeA.begin(), shapeA.begin() + (rankA - 2));
        Shape shapeB1(shapeB.begin(), shapeB.begin() + (rankB - 2));
        Shape ret = infer_broadcast(shapeA1, shapeB1);
        // if (ret.empty()) {
        //     b = 1;
        // } else {
        //     b = std::accumulate(ret.begin(), ret.end(), 1, std::multiplies<int>());
        // }
        // 推导k
        auto kA = *(transA ? shapeA.rbegin() + 1 : shapeA.rbegin());
        auto kB = *(transB ? shapeB.rbegin() : shapeB.rbegin() + 1);
        IT_ASSERT(kA == kB);
        // 有了k后，推导m, n非常简单
        m = *(transA ? shapeA.rbegin() : shapeA.rbegin() + 1);
        n = *(transB ? shapeB.rbegin() + 1 : shapeB.rbegin());
        k = kA;
        ret.emplace_back(m);
        ret.emplace_back(n);
        return {{ret}};
    }

} // namespace infini