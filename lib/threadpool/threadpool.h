/*********************************************************
*
*  Copyright (C) 2014 by Vitaliy Vitsentiy
*  Modified 2021 by Josh Morman
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*********************************************************/

#include <queue>
#include <thread>        
#include <mutex>       
#include <atomic>
#include <future>
#include <iostream>

#include "detail.h"
#include "worker.h"

#include <pmt/pmt.h>

namespace gr {
namespace wifigpu {

#if 1
class threadpool {
public:
  std::vector<std::shared_ptr<std::atomic<bool>>> flags;
  threadpool(unsigned numthreads, unsigned queuedepth) {
    m_numthreads = numthreads;
    m_queuedepth = queuedepth;

    this->threads.resize(numthreads);
    this->flags.resize(numthreads);

    for (int i = 0; i < numthreads; i++) {
      this->flags[i] = std::make_shared<std::atomic<bool>>(false);
      std::shared_ptr<std::atomic<bool>> flag(
          this->flags[i]); // a copy of the shared ptr to the flag
      auto f = [this, i, flag /* a copy of the shared ptr to the flag */]() {
        // std::cout << "entering thread " << i << std::endl;
        std::atomic<bool> &_flag = *flag;
        pmt::pmt_t _p;
        bool isPop = this->q.pop(_p);
        burst_worker b(i);
        while (true) {
          while (isPop) { // if there is anything in the queue
            this->oq.push(b.process(_p));
            if (_flag) {
              std::cout << "return1 " << i << std::endl;
              return; // the thread is wanted to stop, return even if the queue
                      // is not empty yet
            } else
              isPop = this->q.pop(_p);
          }
          // the queue is empty here, wait for the next command
          std::unique_lock<std::mutex> lock(this->m_mutex);
          ++this->nWaiting;
          this->cv.wait(lock, [this, &_p, &isPop, &_flag]() {
            isPop = this->q.pop(_p);
            return isPop || this->isDone || _flag;
          });
          --this->nWaiting;
          // if (!isPop) {
          //     std::cout << "return2 " << i << std::endl;
          //     return;  // if the queue is empty and this->isDone == true or
          //     *flag then return
          // }
        }
        std::cout << "return3 " << i << std::endl;
      };
      this->threads[i].reset(new std::thread(f));
    }
  }
  ~threadpool();
  void enqueue(pmt::pmt_t p) {
    // std::cout << "enqueuing ... " << std::endl;
    this->q.push(p);
    std::unique_lock<std::mutex> lock(this->m_mutex);
    this->cv.notify_one();
  }
  pmt::pmt_t dequeue() {
    pmt::pmt_t r = pmt::PMT_NIL;
    this->oq.pop(r);
    return r;
  }

private:
  std::vector<std::unique_ptr<std::thread>> threads;
  std::thread &get_thread(int i) { return *this->threads[i]; }
  detail::Queue<pmt::pmt_t> q;
  detail::Queue<pmt::pmt_t> oq;
  unsigned m_numthreads;
  unsigned m_queuedepth;
  std::mutex m_mutex;
  std::condition_variable cv;
  std::atomic<bool> isDone;
  std::atomic<bool> isStop;
  std::atomic<int> nWaiting; // how many threads are waiting
};
#endif

} // namespace threadpool
} // namespace gr