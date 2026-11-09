#include "bj/core/material.hpp"
#include <array>
#include <cerrno>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <stdexcept>

namespace bj {
    void install_dealer_network_guard() {

        constexpr int denied[] = {SYS_socket,
                                  SYS_socketpair,
                                  SYS_connect,
                                  SYS_bind,
                                  SYS_listen,
                                  SYS_accept,
                                  SYS_accept4,
                                  SYS_sendto,
                                  SYS_recvfrom,
                                  SYS_sendmsg,
                                  SYS_recvmsg,
                                  SYS_sendmmsg,
                                  SYS_recvmmsg};
        std::array<sock_filter, 5 + 2 * std::size(denied)> code{};
        std::size_t i = 0;
        code[i++] = BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, arch));
        code[i++] = BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0);
        code[i++] = BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS);
        code[i++] = BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr));
        for (int call : denied) {
            code[i++] = BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, static_cast<unsigned>(call), 0, 1);
            code[i++] = BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM);
        }
        code[i++] = BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
        sock_fprog filter{static_cast<unsigned short>(i), code.data()};
        if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) ||
            ::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &filter)) {
            throw std::runtime_error("cannot install mandatory dealer offline network guard");
        }
    }
}
