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

        SparseInput *half1, *half2;
        SparseInput *pawn1, *pawn2;
        DenseInput *psqt, *bucket_selector;

        const size_t n_squares     = 64;
        const size_t n_piece_types = 6;
        const size_t n_colours     = 2;

        const size_t n_features      = n_squares * n_piece_types * n_colours;
        const size_t n_pawn_features = 96;

        // Defines the sizes of the Network's Layers

        const size_t n_l0 = 64; // Outputs, for each half. So L1 input is 2 x n_l0
        const size_t n_l1 = 8;  // Outputs. Makes this layer: 2 x n_l0 by n_l1
        const size_t n_l2 = 16;
        const size_t n_l3 = 1;

        const size_t n_buckets = 8;
        const size_t pawn_n_l0 = 64;

        // Defines miscellaneous hyper-parameters

        const double wdl_percent  = 0.50; // Use x% from the WDL label
        const double eval_percent = 0.50; // Use y% from the EVAL label
        const double sigm_coeff   = 2.315 / 400.00;

        // Defines the mechanism of Quantization

        const size_t quant_ft      = 32;
        const size_t quant_l1      = 32;
        const size_t quant_l2      = 32;
        const size_t quant_l3      = 32;
        const size_t quant_pawn_ft = 32;

        const double clip_ft      = 63.0  / quant_ft;
        const double clip_l1      = 127.0 / quant_l1;
        const double clip_l2      = 127.0 / quant_l2;
        const double clip_l3      = 127.0 / quant_l3;
        const double clip_pawn_ft = 127.0 / quant_pawn_ft;

        // Defines the ADAM Optimizer's hyper-parameters

        const double adam_beta1  = 0.95;
        const double adam_beta2  = 0.999;
        const double adam_eps    = 1e-8;
        const double adam_warmup = 5 * 16384;

        EtherealModel(size_t save_rate = 50) : ChessModel(0) {

            half1 = add<SparseInput>(n_features, 31); // Max 31 relations "kp"
            half2 = add<SparseInput>(n_features, 31); // Max 31 relations "kp"

            pawn1 = add<SparseInput>(n_pawn_features, 16); // Max 16 Pawns ever
            pawn2 = add<SparseInput>(n_pawn_features, 16); // Max 16 Pawns ever

            psqt  = add<DenseInput>(1);
            bucket_selector = add<DenseInput>(1);

            /* Regular FT */

            auto ft  = add<FeatureTransformer>(half1, half2, n_l0);
            ft->ft_regularization  = 1.0 / 16384.0 / 4194304.0;

            auto fta = add<ClippedRelu>(ft);
            fta->max = 127.0 / 32.0;

            auto ft_pair = add<ChunkwiseMul>(fta, 4);

            /* Pawn FT */

            auto pawn_ft = add<FeatureTransformer>(pawn1, pawn2, pawn_n_l0);
            pawn_ft->ft_regularization  = 1.0 / 16384.0 / 4194304.0;

            auto pawn_fta = add<ClippedRelu>(pawn_ft);
            pawn_fta->max = 127.0 / 32.0;

            auto pawn_ft_pair = add<ChunkwiseMul>(pawn_fta, 4);

            // Concat the multiple FTs

            auto final_ft = add<Merge>(ft_pair, pawn_ft_pair);

            // L1 -> L2 -> L3 + PSQT -> Sigmoid

            auto l1  = add<Affine>(final_ft, n_l1);
            auto l1a = add<ReLU>(l1);

            auto l2  = add<AffineMulti>(l1a, n_l2, n_buckets);
            auto l2a = add<ReLU>(l2);

            auto l3  = add<AffineBatched>(l2a, n_l3 * n_buckets, n_buckets);
            auto l3s = add<SelectSingle>(l3, bucket_selector, n_buckets);
            auto l3m = add<WeightedSum>(psqt, l3s, 1, 1);
            auto l3a = add<Sigmoid>(l3m, sigm_coeff);

            set_save_frequency(save_rate);

            add_optimizer(
                AdamWarmup({
                    {OptimizerEntry {&ft->weights}.clamp(-clip_ft, clip_ft)},
                    {OptimizerEntry {&ft->bias}},
                    {OptimizerEntry {&pawn_ft->weights}.clamp(-clip_pawn_ft, clip_pawn_ft)},
                    {OptimizerEntry {&pawn_ft->bias}},
                    {OptimizerEntry {&l1->weights}.clamp(-clip_l1, clip_l1)},
                    {OptimizerEntry {&l1->bias}},
                    {OptimizerEntry {&l2->weights}.clamp(-clip_l2, clip_l2)},
                    {OptimizerEntry {&l2->bias}},
                    {OptimizerEntry {&l3->weights}.clamp(-clip_l3, clip_l3)},
                    {OptimizerEntry {&l3->bias}}
                }, adam_beta1, adam_beta2, adam_eps, adam_warmup)
            );
        }

        int ft_index(chess::Square pc_sq, chess::Piece pc, chess::Color view, chess::Square k_sq) {

            const chess::PieceType piece_type  = chess::type_of(pc);
            const chess::Color     piece_color = chess::color_of(pc);

            // Swap it to the `view`s perspective
            pc_sq = relative_square(view, pc_sq);

            // If the King is on the Queen Side, Mirror the piece
            if (queen_side_sq(relative_square(view, k_sq)))
                pc_sq = mirror_square(pc_sq);

            // If King, map RHS to usual 32 squares
            if (piece_type == chess::KING)
                pc_sq = 4 * rank_of(pc_sq) + file_of(pc_sq) - 4;

            int idx = (piece_color == view) * n_squares * n_piece_types
                    +  piece_type * n_squares
                    +  pc_sq;

            return idx;
        }

        int pawn_index(chess::Square pc_sq, chess::Piece pc, chess::Color view, chess::Square k_sq) {

            const chess::Color piece_color = chess::color_of(pc);

            // Swap it to the `view`s perspective
            pc_sq = relative_square(view, pc_sq);

            // If the King is on the Queen Side, Mirror the piece
            if (queen_side_sq(relative_square(view, k_sq)))
                pc_sq = mirror_square(pc_sq);

            int idx = (piece_color == view) * (n_squares - 16)
                    + (pc_sq - 8);

            return idx;
        }

        void setup_inputs_and_outputs(dataset::DataSet<chess::Position>* positions) {

            half1->sparse_output.clear();
            half2->sparse_output.clear();

            pawn1->sparse_output.clear();
            pawn2->sparse_output.clear();

            auto& target = m_loss->target;

            int material_values[] = {208, 854, 915, 1380, 2682, 0};

            #pragma omp parallel for schedule(static) num_threads(8)
            for (int b = 0; b < positions->header.entry_count; b++) {

                int mat_value = 0;

                chess::Position* pos = &positions->positions[b];
                chess::Color     stm = pos->m_meta.stm();

                chess::BB bb { pos->m_occupancy };

                const chess::Square wking = pos->get_king_square<chess::WHITE>();
                const chess::Square bking = pos->get_king_square<chess::BLACK>();

                const chess::Square stm_king  = stm == chess::WHITE ? wking : bking;
                const chess::Square nstm_king = stm == chess::WHITE ? bking : wking;

                for (int index = 0; bb; index++) {

                    const chess::Square sq = chess::lsb(bb);
                    const chess::Piece  pc = pos->m_pieces.get_piece(index);

                    const chess::PieceType piece_type  = chess::type_of(pc);
                    const chess::Color     piece_color = chess::color_of(pc);

                    // Don't set for the Opposing player's King
                    if (piece_type != chess::KING || piece_color == stm)
                        half1->sparse_output.set(b, ft_index(sq, pc, stm, stm_king));

                    // Don't set for the Opposing player's King
                    if (piece_type != chess::KING || piece_color == !stm)
                        half2->sparse_output.set(b, ft_index(sq, pc, !stm, nstm_king));

                    // Special Pawn net
                    if (piece_type == chess::PAWN) {
                        pawn1->sparse_output.set(b, pawn_index(sq, pc,  stm,  stm_king));
                        pawn2->sparse_output.set(b, pawn_index(sq, pc, !stm, nstm_king));
                    }

                    bb = chess::lsb_reset(bb);

                    mat_value += chess::color_of(pc) == stm ?  material_values[chess::type_of(pc)]
                                                            : -material_values[chess::type_of(pc)];
                }

                psqt->dense_output.values(0, b) = 0.50 * mat_value;
                bucket_selector->dense_output.values(0, b) = (chess::popcount(pos->m_occupancy) - 1) / 4;

                float eval_target = 1.0 / (1.0 + expf(-pos->m_result.score * sigm_coeff));
                float wdl_target  = (pos->m_result.wdl + 1) / 2.0f; // -> [1.0, 0.5, 0.0] WDL

                target(b) = eval_percent * eval_target + wdl_percent * wdl_target;
            }
        }
    };
}