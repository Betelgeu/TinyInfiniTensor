#include "core/graph.h"
#include <algorithm>
#include <memory>
#include <numeric>
#include <queue>
#include "core/common.h"
#include "core/op_type.h"
#include "core/runtime.h"
#include "operators/transpose.h"
#include "operators/matmul.h"

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    bool invalid_trans(const std::vector<int> &permute1, const std::vector<int> &permute2)
    {
        IT_ASSERT(permute1.size() == permute2.size());
        std::vector<int> res(permute1.size());
        for (int i = 0; i < (int)permute1.size(); ++i)
        {
            res[i] = permute2[permute1[i]];
            if(res[i] != i) {
                return false;
            }
        }
        return true;
    }

    bool last2_premute(const std::vector<int> &permute)
    {
        for(int i = 0; i < (int)permute.size() - 2; ++i) {
            if(permute[i] != i) {
                return false;
            }
        }
        return permute.size() >= 2 && permute[permute.size() - 1] == (int)permute.size() - 2 && permute[permute.size() - 2] == (int)permute.size() - 1;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        IT_ASSERT(topo_sort());

        // 注意不能直接在for()中删除节点，否则vector循环会出错
        auto remove_ops = std::vector<Operator>();

        for(auto it = ops.begin(); it != ops.end();) {
            auto jt = it + 1;

            // 1. 去除冗余的算子
            std::shared_ptr<OperatorObj> op = *it, pre_op = nullptr;
            if(op->getPredecessors().size() != 1) {
            } else if(auto pre_op = op->getPredecessors()[0];
                pre_op->getOpType() != OpType::Transpose || op->getOpType() != OpType::Transpose
            ) {
            } else if(TransposeObj *trans_op = static_cast<TransposeObj*>(op.get()), *trans_pre_op = static_cast<TransposeObj*>(pre_op.get());
                invalid_trans(trans_op->getPermute(), trans_pre_op->getPermute())) {
                // 实际删除这两个transpose需要一堆操作, tensor和op中都是有一堆冗余数据的(前驱和后继)
                // tensor
                auto input_tensor = pre_op->getInputs()[0];
                auto output_tensor = op->getOutputs()[0];
                input_tensor->removeTarget(pre_op);
                for(auto &op : output_tensor->getTargets()) {
                    op->replaceInput(output_tensor, input_tensor);
                    input_tensor->addTarget(op);
                }
                this->removeTensor(output_tensor);
                this->removeTensor(op->getInputs()[0]);
                // op
                std::shared_ptr<OperatorObj> pre_pre_op = nullptr;
                if (pre_op->getPredecessors().size() == 1) { // pre_pre_op不一定存在
                    pre_pre_op = pre_op->getPredecessors()[0];
                    pre_pre_op->removeSuccessors(pre_op);
                }

                auto post_ops = op->getSuccessors();
                for(auto &post_op : post_ops) {
                    post_op->removePredecessors(op);
                    if (pre_pre_op) {
                        pre_pre_op->addSuccessors(post_op);
                    }
                }
                remove_ops.push_back(pre_op);
                remove_ops.push_back(op);
            } else {
            }

            // 2. 合并算子
            if(op->getOpType() == OpType::MatMul) {
                auto matmul_op = static_cast<MatmulObj*>(op.get());
                for(int i = 0; i < 2; i++) { // input vec一定是两个
                    auto input = matmul_op->getInputs()[i];
                    auto pre_op = input->getSource();
                    if(pre_op == nullptr) {
                        continue;
                    } else if(pre_op->getOpType() == OpType::Transpose && pre_op->getSuccessors().size() == 1) {
                        auto trans_pre_op = static_cast<TransposeObj*>(pre_op.get());
                        if (last2_premute(trans_pre_op->getPermute())) {
                            if(i == 0) {
                                matmul_op->setTransA(!matmul_op->getTransA());
                            } else {
                                matmul_op->setTransB(!matmul_op->getTransB());
                            }
                            // tensor
                            auto input_tensor = pre_op->getInputs()[0];
                            input_tensor->removeTarget(pre_op);
                            for(auto &op : input->getTargets()) {
                                op->replaceInput(input, input_tensor);
                                input_tensor->addTarget(op);
                            }
                            this->removeTensor(pre_op->getOutputs()[0]);
                            // op
                            std::shared_ptr<OperatorObj> pre_pre_op = nullptr;
                            if (pre_op->getPredecessors().size() == 1) { // pre_pre_op不一定存在
                                pre_pre_op = pre_op->getPredecessors()[0];
                                pre_pre_op->removeSuccessors(pre_op);
                                pre_pre_op->addSuccessors(op);
                            }
                            matmul_op->removePredecessors(pre_op);
                            remove_ops.push_back(pre_op);
                        }
                    }
                }
            }

            it = jt;
        }

        for(auto &op : remove_ops) {
            this->removeOperator(op);
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        size_t total_size = 0;
        for(auto &tensor : tensors) {
            total_size += tensor->getBytes();
        }
        size_t addr = allocator.alloc(total_size);
        for(auto &tensor : tensors) {
            auto ptr = static_cast<char*>(allocator.getPtr()) + addr;
            tensor->setDataBlob(make_ref<BlobObj>(this->runtime, ptr));
            addr += tensor->getBytes();
        }
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini