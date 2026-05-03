/*
 * CF Framework
 * Copyright (C) 2026 Orion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "SECURITY/cf_aes.h"

#include <string.h>

static cf_u8 cf_aes_g8_mul_mod(cf_u8 p, cf_u8 q)
{
  cf_u8 res = 0;
  do
  {
    if(q & 0x01) res ^= p;
    if(p & 0x80) p = (p << 1) ^ 0x1B;
    else p <<= 1;
  } while(q >>= 1);
  return res;
}

/*
 * Apply the AES S-box to each byte of a key-schedule word.
 */
static cf_u32 cf_aes_sub_word(cf_u32 word)
{
  return ((cf_u32)CF_AES_SBOX[(word >> 0x18) & 0xFF] << 0x18)|
         ((cf_u32)CF_AES_SBOX[(word >> 0x10) & 0xFF] << 0x10)|
         ((cf_u32)CF_AES_SBOX[(word >> 0x08) & 0xFF] << 0x08)|
         ((cf_u32)CF_AES_SBOX[word & 0xFF]);
}

/*
 * AES key-schedule core: rotate, substitute, and mix in the round constant.
 */
static cf_u32 cf_aes_g_func(cf_u32 word, cf_u8 round_i)
{
  cf_u8 tmp[] = {word >> 0x10, word >> 0x08, word, word >> 0x18};
  return (((cf_u32)(CF_AES_SBOX[tmp[0]] ^ CF_AES_RCJ[round_i])) << 0x18)|
          ((cf_u32)CF_AES_SBOX[tmp[1]] << 0x10)|
          ((cf_u32)CF_AES_SBOX[tmp[2]] << 0x08)|
          ((cf_u32)CF_AES_SBOX[tmp[3]]);
}

/*
 * Expand the caller key into all AES round keys stored in cf_aes.
 */
static void cf_aes_key_expansion(cf_aes *aes, const cf_u8 key[CF_AES_MAX_ROUND_KEYS * 4], cf_aes_key_size key_size)
{
  cf_u8 row_size = key_size / 4;
  aes->key_size = key_size; // 16 24 32 bytes
  aes->rounds = row_size + 6; // 10 12 14 rounds
  aes->words = key_size + 28; // 44 52 60 words
  for (cf_u8 i = 0; i < row_size; i++)
  {
    aes->round_keys[i] = 0;
    for (size_t j = 0; j < 4; j++)
    {
      aes->round_keys[i] |= ((cf_u32)key[i * 4 + j] << ((3 - j) * 8));
    }
  }
  for (cf_u8 i = row_size; i < aes->words; i++)
  {
    cf_u32 tmp = aes->round_keys[i - 1];

    if(i % row_size == 0)
      tmp = cf_aes_g_func(tmp, i / row_size);
    else if (row_size == 8 && i % 8 == 4)
      tmp = cf_aes_sub_word(tmp);

    aes->round_keys[i] = aes->round_keys[i - row_size] ^ tmp;
  }
}

/*
 * Initialize an AES context for 128, 192, or 256-bit keys.
 */
cf_status cf_aes_init(cf_aes *aes, const cf_u8 key[CF_AES_MAX_ROUND_KEYS * 4], cf_aes_key_size key_size)
{
  if(aes == CF_NULL || key == CF_NULL) return CF_ERR_NULL;
  if(key_size != CF_AES_KEY_128 && key_size != CF_AES_KEY_192 && key_size != CF_AES_KEY_256) 
    return CF_ERR_INVALID;
  cf_aes_key_expansion(aes, key, key_size);
  return CF_OK;
}

/*
 * Forward AES SubBytes transform.
 */
static void cf_aes_sub_bytes(cf_u8 state[4][4])
{
  for (cf_u8 i = 0; i < 4; i++)
    for (size_t j = 0; j < 4; j++)
      state[i][j] = CF_AES_SBOX[state[i][j]];
}

/*
 * Inverse AES SubBytes transform.
 */
static void cf_aes_inv_sub_bytes(cf_u8 state[4][4])
{
  for (cf_u8 i = 0; i < 4; i++)
    for (size_t j = 0; j < 4; j++)
      state[i][j] = CF_AES_INV_SBOX[state[i][j]];
}

/*
 * Forward AES ShiftRows transform on the 4x4 state matrix.
 */
static void cf_aes_shift_rows(cf_u8 state[4][4])
{
  const cf_u8 tmp[4][4] =
  {
    {state[0][0], state[0][1], state[0][2], state[0][3]},
    {state[1][1], state[1][2], state[1][3], state[1][0]},
    {state[2][2], state[2][3], state[2][0], state[2][1]},
    {state[3][3], state[3][0], state[3][1], state[3][2]},
  };
  for (cf_u8 i = 0; i < 4; i++)
    memcpy(state[i], tmp[i], sizeof (cf_u8) * 4);
}

/*
 * Inverse AES ShiftRows transform on the 4x4 state matrix.
 */
static void cf_aes_inv_shift_rows(cf_u8 state[4][4])
{
  const cf_u8 tmp[4][4] =
  {
    {state[0][0], state[0][1], state[0][2], state[0][3]},
    {state[1][3], state[1][0], state[1][1], state[1][2]},
    {state[2][2], state[2][3], state[2][0], state[2][1]},
    {state[3][1], state[3][2], state[3][3], state[3][0]},
  };
  for (cf_u8 i = 0; i < 4; i++)
    memcpy(state[i], tmp[i], sizeof (cf_u8) * 4);
}

/*
 * Forward AES MixColumns transform using GF(2^8) multiplication from aes.
 */
static void cf_aes_mix_columns(cf_u8 state[4][4])
{
  for (cf_u8 i = 0; i < 4; i++)
  {
      cf_u8 tmp[]= 
      {
        cf_aes_g8_mul_mod(state[0][i], CF_AES_MIX_COLUMN[0][0])^
        cf_aes_g8_mul_mod(state[1][i], CF_AES_MIX_COLUMN[0][1])^
        state[2][i]^
        state[3][i],

        state[0][i]^
        cf_aes_g8_mul_mod(state[1][i], CF_AES_MIX_COLUMN[1][1])^
        cf_aes_g8_mul_mod(state[2][i], CF_AES_MIX_COLUMN[1][2])^
        state[3][i],

        state[0][i]^
        state[1][i]^
        cf_aes_g8_mul_mod(state[2][i], CF_AES_MIX_COLUMN[2][2])^
        cf_aes_g8_mul_mod(state[3][i], CF_AES_MIX_COLUMN[2][3]),

        cf_aes_g8_mul_mod(state[0][i], CF_AES_MIX_COLUMN[3][0])^
        state[1][i]^
        state[2][i]^
        cf_aes_g8_mul_mod(state[3][i], CF_AES_MIX_COLUMN[3][3]),
      };
      state[0][i] = tmp[0];
      state[1][i] = tmp[1];
      state[2][i] = tmp[2];
      state[3][i] = tmp[3];
  }
}

/*
 * Inverse AES MixColumns transform using the inverse coefficient table.
 */
static void cf_aes_inv_mix_columns(cf_u8 state[4][4])
{
  for (cf_u8 i = 0; i < 4; i++)
  {
      cf_u8 tmp[]= 
      {
        cf_aes_g8_mul_mod(state[0][i], CF_AES_INV_MIX_COLUMN[0][0])^
        cf_aes_g8_mul_mod(state[1][i], CF_AES_INV_MIX_COLUMN[0][1])^
        cf_aes_g8_mul_mod(state[2][i], CF_AES_INV_MIX_COLUMN[0][2])^
        cf_aes_g8_mul_mod(state[3][i], CF_AES_INV_MIX_COLUMN[0][3]),

        cf_aes_g8_mul_mod(state[0][i], CF_AES_INV_MIX_COLUMN[1][0])^
        cf_aes_g8_mul_mod(state[1][i], CF_AES_INV_MIX_COLUMN[1][1])^
        cf_aes_g8_mul_mod(state[2][i], CF_AES_INV_MIX_COLUMN[1][2])^
        cf_aes_g8_mul_mod(state[3][i], CF_AES_INV_MIX_COLUMN[1][3]),

        cf_aes_g8_mul_mod(state[0][i], CF_AES_INV_MIX_COLUMN[2][0])^
        cf_aes_g8_mul_mod(state[1][i], CF_AES_INV_MIX_COLUMN[2][1])^
        cf_aes_g8_mul_mod(state[2][i], CF_AES_INV_MIX_COLUMN[2][2])^
        cf_aes_g8_mul_mod(state[3][i], CF_AES_INV_MIX_COLUMN[2][3]),

        cf_aes_g8_mul_mod(state[0][i], CF_AES_INV_MIX_COLUMN[3][0])^
        cf_aes_g8_mul_mod(state[1][i], CF_AES_INV_MIX_COLUMN[3][1])^
        cf_aes_g8_mul_mod(state[2][i], CF_AES_INV_MIX_COLUMN[3][2])^
        cf_aes_g8_mul_mod(state[3][i], CF_AES_INV_MIX_COLUMN[3][3]),
      };
      state[0][i] = tmp[0];
      state[1][i] = tmp[1];
      state[2][i] = tmp[2];
      state[3][i] = tmp[3];
  }
}

/*
 * XOR one round-key word set into the AES state matrix.
 */
static void cf_aes_add_round_key(cf_u8 state[4][4], cf_u32 word[4])
{
  for (cf_u8 i = 0; i < 4; i++)
  {
    state[0][i] = state[0][i] ^ (cf_u8)(word[i] >> 0x18);
    state[1][i] = state[1][i] ^ (cf_u8)(word[i] >> 0x10);
    state[2][i] = state[2][i] ^ (cf_u8)(word[i] >> 0x08);
    state[3][i] = state[3][i] ^ (cf_u8)(word[i]);
  }
}

/*
 * Encrypt one 16-byte block using an initialized AES context.
 */
void cf_aes_encrypt_block(cf_aes *aes, cf_u8 dst[CF_AES_BLOCK_SIZE], const cf_u8 src[CF_AES_BLOCK_SIZE])
{
  if(aes == CF_NULL || dst == CF_NULL || src == CF_NULL) return;

  cf_u8 state[4][4] =
  {
    {src[0], src[4], src[8],  src[12]},
    {src[1], src[5], src[9],  src[13]},
    {src[2], src[6], src[10], src[14]},
    {src[3], src[7], src[11], src[15]},
  };
  cf_aes_add_round_key(state, (cf_u32[]) {aes->round_keys[0], aes->round_keys[1], aes->round_keys[2], aes->round_keys[3]});
  for (cf_u8 i = 1; i < aes->rounds; i++)
  {
    cf_aes_sub_bytes(state);
    cf_aes_shift_rows(state);
    cf_aes_mix_columns(state);
    cf_aes_add_round_key(state,(cf_u32[]) {aes->round_keys[4 * i], aes->round_keys[4 * i + 1], aes->round_keys[4 * i + 2], aes->round_keys[4 * i + 3]});
  }
  cf_aes_sub_bytes(state);
  cf_aes_shift_rows(state);
  cf_aes_add_round_key(state,(cf_u32[]) {aes->round_keys[4 * aes->rounds], aes->round_keys[4 * aes->rounds + 1], aes->round_keys[4 * aes->rounds + 2], aes->round_keys[4 * aes->rounds + 3]});
  for (cf_u8 i = 0; i < 4; i++)
    for (cf_u8 j = 0; j < 4; j++)
      dst[j * 4 + i] = state[i][j];
}

/*
 * Decrypt one 16-byte block using an initialized AES context.
 */
void cf_aes_decrypt_block(cf_aes *aes, cf_u8 dst[CF_AES_BLOCK_SIZE], const cf_u8 src[CF_AES_BLOCK_SIZE])
{
  if(aes == CF_NULL || dst == CF_NULL || src == CF_NULL) return;

  cf_u8 state[4][4] =
  {
    {src[0], src[4], src[8],  src[12]},
    {src[1], src[5], src[9],  src[13]},
    {src[2], src[6], src[10], src[14]},
    {src[3], src[7], src[11], src[15]},
  };
  cf_aes_add_round_key(state, (cf_u32[]) {aes->round_keys[4 * aes->rounds], aes->round_keys[4 * aes->rounds + 1], aes->round_keys[4 * aes->rounds + 2], aes->round_keys[4 * aes->rounds + 3]});
  for (cf_u8 i = 1; i < aes->rounds; i++)
  {
    cf_aes_inv_shift_rows(state);
    cf_aes_inv_sub_bytes(state);
    cf_aes_add_round_key(state, (cf_u32[]) {aes->round_keys[4 * (aes->rounds - i)], aes->round_keys[4 * (aes->rounds - i) + 1], aes->round_keys[4 * (aes->rounds - i) + 2], aes->round_keys[4 * (aes->rounds - i) + 3]});
    cf_aes_inv_mix_columns(state);
  }
  cf_aes_inv_shift_rows(state);
  cf_aes_inv_sub_bytes(state);
  cf_aes_add_round_key(state, (cf_u32[]) {aes->round_keys[0], aes->round_keys[1], aes->round_keys[2], aes->round_keys[3]});
  for (cf_u8 i = 0; i < 4; i++)
    for (cf_u8 j = 0; j < 4; j++)
      dst[j * 4 + i] = state[i][j];
}

/*
 * Apply PKCS#7 padding to a mutable byte buffer for AES block processing.
 */
cf_status cf_aes_pkcs7_pad(cf_buffer *buffer)
{
  if(buffer == CF_NULL) return CF_ERR_NULL;
  if(cf_buffer_is_valid(buffer) == CF_FALSE) return CF_ERR_STATE;

  cf_usize pad_len = CF_AES_BLOCK_SIZE - buffer->len % CF_AES_BLOCK_SIZE;
  cf_u8 byte[pad_len];
  memset(byte, pad_len, pad_len);
  cf_bytes bytes = (cf_bytes) {.data = byte, .elem_size = sizeof (cf_u8), .len = pad_len};
  return cf_buffer_append_bytes(buffer, bytes);
}

/*
 * Validate and remove PKCS#7 padding after AES block processing.
 */
cf_status cf_aes_pkcs7_unpad(cf_buffer *buffer)
{
  if(buffer == CF_NULL) return CF_ERR_NULL;
  if(cf_buffer_is_valid(buffer) == CF_FALSE) return CF_ERR_STATE;
  if(buffer->data == CF_NULL || buffer->len == 0 || buffer->len % CF_AES_BLOCK_SIZE != 0)
    return CF_ERR_INVALID_PADDING;

  cf_u8 pad = buffer->data[buffer->len - 1];

  if (pad == 0 || pad > CF_AES_BLOCK_SIZE || pad > buffer->len)
    return CF_ERR_INVALID_PADDING;

  for (cf_usize i = 0; i < pad; i++)
  {
    if (buffer->data[buffer->len - 1 - i] != pad)
      return CF_ERR_INVALID_PADDING;
  }

  buffer->len -= pad;
  return CF_OK;
}
