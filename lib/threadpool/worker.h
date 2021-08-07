#include <volk/volk.h>

#include <vector>

using namespace std;

namespace gr {
namespace wifigpu {

class burst_worker {
private:
  int m_id;
  int packet_cnt;

public:
  burst_worker(int id) {
    m_id = id;
    packet_cnt = 0;
  }

  ~burst_worker() {
  }

  pmt::pmt_t process(pmt::pmt_t msg) {
    pmt::pmt_t meta(pmt::car(msg));
    pmt::pmt_t data(pmt::cdr(msg));

    size_t num_pdu_samples = pmt::length(data);
    size_t len_bytes(0);

    const void *samples = NULL;
    // supported PMT vector types -- move this somewhere else later
    if (pmt::is_u8vector(data)) {
      samples = (const void *)pmt::u8vector_elements(data, len_bytes);
    } else if (pmt::is_s8vector(data)) {
      samples = (const void *)pmt::s8vector_elements(data, len_bytes);
    } else if (pmt::is_u16vector(data)) {
      samples = (const void *)pmt::u16vector_elements(data, len_bytes);
    } else if (pmt::is_s16vector(data)) {
      samples = (const void *)pmt::s16vector_elements(data, len_bytes);
    } else if (pmt::is_u32vector(data)) {
      samples = (const void *)pmt::u32vector_elements(data, len_bytes);
    } else if (pmt::is_s32vector(data)) {
      samples = (const void *)pmt::s32vector_elements(data, len_bytes);
    } else if (pmt::is_u64vector(data)) {
      samples = (const void *)pmt::u64vector_elements(data, len_bytes);
    } else if (pmt::is_s64vector(data)) {
      samples = (const void *)pmt::s64vector_elements(data, len_bytes);
    } else if (pmt::is_f32vector(data)) {
      samples = (const void *)pmt::f32vector_elements(data, len_bytes);
    } else if (pmt::is_f64vector(data)) {
      samples = (const void *)pmt::f64vector_elements(data, len_bytes);
    } else if (pmt::is_c32vector(data)) {
      samples = (const void *)pmt::c32vector_elements(data, len_bytes);
    } else if (pmt::is_c64vector(data)) {
      samples = (const void *)pmt::c64vector_elements(data, len_bytes);
    }

    if (!samples) {
      std::cout << "ERROR: Invalid input type - must be a PMT vector"
                << std::endl;
      return pmt::PMT_NIL;
    }

    // make sure type matches with input signature
    // TODO
    int nproduced = num_pdu_samples;


    // Insert MAC Decode code here
    std::cout << "Threadpool got new burst" << std::endl;

    // repackage as pmt and place on output queue
    // pmt::pmt_t vecpmt(pmt::init_c32vector(nproduced, &buffer1[0]));
    // pmt::pmt_t pdu(pmt::cons(pmt::PMT_NIL, vecpmt));
    pmt::pmt_t pdu = nullptr;
    
    this->packet_cnt++;
    // std::cout << "worker: " << packet_cnt << std::endl;

    return pdu;
  }
};
} // namespace metablocks
} // namespace gr