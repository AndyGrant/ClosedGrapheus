#pragma once

#include <utility>

#include "chessmodel.h"

namespace {

    int file_of(int sq)            { return sq % 8;          }
    int rank_of(int sq)            { return sq / 8;          }
    int square(int rank, int file) { return rank * 8 + file; }

    int relative_rank_of(int c, int sq) { return c == chess::WHITE ? rank_of(sq) : 7 - rank_of(sq); }
    int relative_square(int c, int sq)  { return square(relative_rank_of(c, sq), file_of(sq));      }
    int mirror_square(int sq)           { return square(rank_of(sq), 7 - file_of(sq));              }

    int queen_side_sq(int sq) {
        return (0x0F0F0F0F0F0F0F0FULL >> sq) & 1;
    }

    int sq64_to_sq32(int sq) {

        static const int LUT[] = {
             3,  2,  1,  0,  0,  1,  2,  3,
             7,  6,  5,  4,  4,  5,  6,  7,
            11, 10,  9,  8,  8,  9, 10, 11,
            15, 14, 13, 12, 12, 13, 14, 15,
            19, 18, 17, 16, 16, 17, 18, 19,
            23, 22, 21, 20, 20, 21, 22, 23,
            27, 26, 25, 24, 24, 25, 26, 27,
            31, 30, 29, 28, 28, 29, 30, 31,
        };

        return LUT[sq];
    }
}

namespace model {

    struct EtherealModel : ChessModel {

        SparseInput *halfkp1, *halfkp2;

        // Defines the halfkp + virtual PSQT realtions

        const size_t n_king_buckets  = 32;
        const size_t n_relations     = 10;
        const size_t n_squares       = 64;

        const size_t n_real_features    = n_king_buckets * n_relations * n_squares;
        const size_t n_virtual_features = n_relations * n_squares;
        const size_t n_features         = n_real_features + n_virtual_features;

        // Defines the sizes of the Network's Layers

        const size_t n_l0 = 768;
        const size_t n_l1 = 8;
        const size_t n_l2 = 32;
        const size_t n_l3 = 1;

        // Defines miscellaneous hyper-parameters

        const double wdl_percent  = 0.50;
        const double eval_percent = 0.50;
        const double sigm_coeff   = 2.315 / 400.00;

        // Defines the mechanism of Quantization

        const size_t quant_ft = 64; // x64
        const size_t quant_l1 = 32; // x32
        const size_t quant_l2 = 1;  // None
        const size_t quant_l3 = 1;  // None

        const double clip_l1  = 127.0 / quant_l1;

        // Defines the ADAM Optimizer's hyper-parameters

        const double adam_beta1  = 0.95;
        const double adam_beta2  = 0.999;
        const double adam_eps    = 1e-8;
        const double adam_warmup = 5 * 16384;

        EtherealModel(size_t save_rate = 50) : ChessModel(0) /* TODO: Fix unneeded Lambda */ {

            halfkp1 = add<SparseInput>(n_features, 60); // Real + Virtual
            halfkp2 = add<SparseInput>(n_features, 60); // Real + Virtual

            auto ft  = add<FeatureTransformer>(halfkp1, halfkp2, n_l0);
            ft->ft_regularization  = 1.0 / 16384.0 / 4194304.0;

            auto fta = add<ClippedRelu>(ft);
            fta->max = 127.0;

            auto l1  = add<Affine>(fta, n_l1);
            auto l1a = add<ReLU>(l1);

            auto l2  = add<Affine>(l1a, n_l2);
            auto l2a = add<ReLU>(l2);

            auto l3  = add<Affine>(l2a, n_l3);
            auto l3a = add<Sigmoid>(l3, sigm_coeff);

            set_save_frequency(save_rate);

            add_optimizer(
                AdamWarmup({
                    {OptimizerEntry {&ft->weights}}, {OptimizerEntry {&ft->bias}},
                    {OptimizerEntry {&l1->weights}.clamp(-clip_l1, clip_l1)}, {OptimizerEntry {&l1->bias}},
                    {OptimizerEntry {&l2->weights}}, {OptimizerEntry {&l2->bias}},
                    {OptimizerEntry {&l3->weights}}, {OptimizerEntry {&l3->bias}}
                }, adam_beta1, adam_beta2, adam_eps, adam_warmup)
            );

            add_quantization(Quantizer{
                "quant", save_rate,
                QuantizerEntry<int16_t>(&ft->weights.values, quant_ft),
                QuantizerEntry<int16_t>(&ft->bias.values,    quant_ft),
                QuantizerEntry<int8_t >(&l1->weights.values, quant_l1),
                QuantizerEntry<int32_t>(&l1->bias.values,    quant_l1),
                QuantizerEntry<float  >(&l2->weights.values, quant_l2),
                QuantizerEntry<float  >(&l2->bias.values,    quant_l2),
                QuantizerEntry<float  >(&l3->weights.values, quant_l3),
                QuantizerEntry<float  >(&l3->bias.values,    quant_l3),
            });
        }

        std::pair<int, int> ft_index(chess::Square pc_sq, chess::Piece pc, chess::Square k_sq, chess::Color view) {

            chess::PieceType piece_type  = chess::type_of(pc);
            chess::Color     piece_color = chess::color_of(pc);

            chess::Square rel_k_sq  = relative_square(view, k_sq);
            chess::Square rel_pc_sq = relative_square(view, pc_sq);

            if (queen_side_sq(rel_k_sq))
                rel_k_sq = mirror_square(rel_k_sq), rel_pc_sq = mirror_square(rel_pc_sq);

            int ft_real = n_squares * n_relations * sq64_to_sq32(rel_k_sq)
                        + n_squares * (5 * (view == piece_color) + piece_type)
                        + rel_pc_sq;

            int ft_virtual = n_squares * (5 * (view == piece_color) + piece_type)
                           + rel_pc_sq;

            return std::make_pair(ft_real, ft_virtual + n_real_features);
        }

        void setup_inputs_and_outputs(dataset::DataSet<chess::Position>* positions) {

            halfkp1->sparse_output.clear();
            halfkp2->sparse_output.clear();

            auto& target = m_loss->target;

            #pragma omp parallel for schedule(static) num_threads(4)
            for (int b = 0; b < positions->header.entry_count; b++) {

                chess::Position* pos = &positions->positions[b];
                chess::Color     stm = pos->m_meta.stm();

                chess::Square wking = pos->get_king_square<chess::WHITE>();
                chess::Square bking = pos->get_king_square<chess::BLACK>();

                chess::Square stm_king  = stm == chess::WHITE ? wking : bking;
                chess::Square nstm_king = stm == chess::WHITE ? bking : wking;

                chess::BB bb { pos->m_occupancy };

                for (int index = 0; bb; index++) {

                    chess::Square sq = chess::lsb(bb);
                    chess::Piece  pc = pos->m_pieces.get_piece(index);

                    if (chess::type_of(pc) != chess::KING) {

                        auto [stm_real,  stm_virtual ] = ft_index(sq, pc, stm_king, stm);
                        auto [nstm_real, nstm_virtual] = ft_index(sq, pc, nstm_king, !stm);

                        halfkp1->sparse_output.set(b, stm_real);
                        halfkp1->sparse_output.set(b, stm_virtual);

                        halfkp2->sparse_output.set(b, nstm_real);
                        halfkp2->sparse_output.set(b, nstm_virtual);
                    }

                    bb = chess::lsb_reset(bb);
                }

                float eval_target = 1.0 / (1.0 + expf(-pos->m_result.score * sigm_coeff));
                float wdl_target  = (pos->m_result.wdl + 1) / 2.0f; // -> [1.0, 0.5, 0.0] WDL

                target(b) = eval_percent * eval_target + wdl_percent * wdl_target;
            }
        }
    };
}