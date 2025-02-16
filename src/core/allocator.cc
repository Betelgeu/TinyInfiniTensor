#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        // 遍历freeBlocks，找到第一个满足条件的block即可
        for(auto it = freeBlocks.begin(); it != freeBlocks.end(); it++) {
            if(it->second >= size) {
                size_t addr = it->first;
                size_t remain = it->second - size;
                freeBlocks.erase(it);
                if (remain > 0) {
                    freeBlocks[addr + size] = remain;
                }
                used += size;
                peak = std::max(peak, used);
                return addr;
            }
        }
        // 如果没有找到合适的block，抛出异常
        throw std::runtime_error("No enough memory");
        return 0;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        size_t lhs = addr;
        size_t rhs = addr + size;

        auto it = freeBlocks.lower_bound(addr);
        if (it != freeBlocks.begin()) {
            auto jt = it--;
            if (jt->first + jt->second >= addr) {
                lhs = jt->first;
                this->used += jt->second;
                freeBlocks.erase(jt);
            }
        }
        while (it != freeBlocks.end() && it->first <= addr + size) {
            rhs = std::max(it->first + it->second, rhs);
            auto jt = it; jt++;
            this->used += it->second;
            freeBlocks.erase(it);
            it = jt;
        }
        
        freeBlocks[lhs] = rhs - lhs;
        this->used -= (rhs - lhs);

    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
