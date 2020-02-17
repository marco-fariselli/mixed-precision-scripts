#my script
from data import get_dataset, get_num_classes
from preprocess import get_transform
from utils import *
import models
import quantization

def generate_Z(m,lay,param_list):
    if lay == 1:
        IN_Z = 127
    else:
        IN_Z = 0

    OUT_Z = 0

    if param_list[lay-1]['quant_type'] == 'PerLayerAsymPACT':
        str_z = '#define CONV'+ str(lay) + '_W_Z			('+ str(int(- m[2*(lay-1)].weight.data.min()))+ ')'
        str_z += '\n'
        str_z += '#define CONV' + str(lay) + '_IN_Z			('+str(IN_Z)+')' 
        str_z += '\n'
        str_z += '#define CONV' + str(lay) + '_OUT_Z		('+str(OUT_Z)+')'
        str_z += '\n'
    elif param_list[lay-1]['quant_type'] == 'PerChannelsAsymMinMax':
        str_z = 'static const int16_t CONV'+ str(lay) + '_W_Z[] ={'
        for i in range(m[2*(lay-1)].weight.size(0)):
            str_z += str(int(- m[2*(lay-1)].weight.data[i][:][:][:].min())) + ', ' 
        str_z += '\\n'
        str_z = str_z[:-4]+'};'
        str_z += '\n'
        str_z += '#define CONV' + str(lay) + '_IN_Z			('+str(IN_Z)+')'
        str_z += '\n'
        str_z += '#define CONV' + str(lay) + '_OUT_Z		('+str(OUT_Z)+')'
        str_z += '\n'
    return str_z
        
def generate_M0_N0(m,lay,param_list):
    if param_list[lay-1]['fold_type'] == 'ICN':
        str_m0= 'static const int32_t CONV' + str(lay) + '_M_ZERO[] =  {'
        for i in range(m[2*lay-1].M_ZERO.size(0)):
            str_m0 += str(int(2**31 * m[2*lay-1].M_ZERO[i].item())) + ', '
        str_m0 += '\\\n'
        str_m0 = str_m0[:-4]+'};'

        str_n0 = 'static const int8_t CONV'+ str(lay) +'_N_ZERO[] = {'
        for i in range(m[2*lay-1].N_ZERO.size(0)):
            str_n0 += str(int(-math.log2(m[2*lay-1].N_ZERO[i])-1)) + ', '
        str_n0 += '\\\n'
        str_n0 = str_n0[:-4] + '};'
                          
    elif param_list[lay-1]['fold_type'] == 'folding_weights':
        str_m0 = '#define CONV'+ str(lay) +'_M_ZERO			('+ str(int(2**31 *m[2*lay-1].M_ZERO))+')'
        str_n0 = '#define CONV'+ str(lay) +'_N_ZERO			('+ str(int(-m[2*lay-1].N_ZERO)-1)+')'
    return str_m0 + '\n' + str_n0
            
def generate_bias(m,lay):
    str_bias = '#define CONV'+ str(lay) +'_BIAS {'
    for i in range(m[(lay-1)*2].bias.data.size(0)):
        str_bias += str(int(m[(lay-1)*2].bias.data[i])) + ', '
    str_bias += '\\\n'
    str_bias = str_bias[:-4] + '}'
    return str_bias

def generate_weights_PL_wt8bit(m,lay, act_input_bits, param_list):
    str_wt = '#define CONV'+ str(lay)+ '_WT {\\\n' 
    wt_bits = param_list[lay-1]['w_bits']
    Z_wt = - m[2*(lay-1)].weight.data.min() 
    biased_wt = (m[2*(lay-1)].weight.data + Z_wt)
    if wt_bits == 8: #order of weights depends on the bitwise
        if param_list[lay-1]['quant_conv'].groups == 1: #order is different if the layer function is depthwise (groups!=1) or convolve (odd layers)
            if act_input_bits == 8:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(m[2*(lay-1)].weight.data.size(1)):
                                str_wt += str(int(biased_wt[i][p][j][k])) + ', '
                            if lay == 1: #the first layer has to be zero padded
                                str_wt += str(int(Z_wt)) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt
            
            elif act_input_bits == 4:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(0, m[2*(lay-1)].weight.data.size(1), 8):
                                str_wt += str(int(biased_wt[i][p][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+1][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+4][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+5][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+2][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+3][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+6][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+7][j][k])) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt

            elif act_input_bits == 2:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(0, m[2*(lay-1)].weight.data.size(1), 16):
                                str_wt += str(int(biased_wt[i][p][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+1][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+8][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+9][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+2][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+3][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+10][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+11][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+4][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+5][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+12][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+13][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+6][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+7][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+14][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+15][j][k])) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt

        else:  #depthwise
            for i in range(m[2*(lay-1)].weight.size(1)):
                for j in range(m[2*(lay-1)].weight.data.size(2)):
                    for k in range(m[2*(lay-1)].weight.data.size(2)):
                        for p in range(m[2*(lay-1)].weight.data.size(0)):
                            str_wt += str(int(biased_wt[p][i][j][k])) + ', '
                        str_wt += '\\\n'
            str_wt = str_wt[:-4]+'}'
            return str_wt
                                              
                                      
def generate_weights_PL_wt4bit(m,lay,act_input_bits,param_list):
    str_wt = '#define CONV'+ str(lay)+ '_WT {\\\n' 
    wt_bits = param_list[lay-1]['w_bits']
    Z_wt = - m[2*(lay-1)].weight.data.min() 
    biased_wt = (m[2*(lay-1)].weight.data + Z_wt)
    if wt_bits == 4:     
        if param_list[lay-1]['quant_conv'].groups == 1: #convolve
            if act_input_bits == 8:            
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(0, m[2*(lay-1)].weight.data.size(1)-7, 8):
                                w0w1 = concat4bit(int(biased_wt[i][p+1][j][k]),int(biased_wt[i][p][j][k]))
                                w4w5 = concat4bit(int(biased_wt[i][p+5][j][k]),int(biased_wt[i][p+4][j][k]))
                                w2w3 = concat4bit(int(biased_wt[i][p+3][j][k]),int(biased_wt[i][p+2][j][k]))
                                w6w7 = concat4bit(int(biased_wt[i][p+7][j][k]),int(biased_wt[i][p+6][j][k]))
                                str_wt += str(w0w1) + ', ' + str(w4w5)+ ', ' + str(w2w3) + ', ' + str(w6w7) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt

            elif act_input_bits == 4:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(0, m[2*(lay-1)].weight.data.size(1)-7, 8):
                                w0w1 = concat4bit(int(biased_wt[i][p+1][j][k]),int(biased_wt[i][p][j][k]))
                                w2w3 = concat4bit(int(biased_wt[i][p+3][j][k]),int(biased_wt[i][p+2][j][k]))
                                w4w5 = concat4bit(int(biased_wt[i][p+5][j][k]),int(biased_wt[i][p+4][j][k]))
                                w6w7 = concat4bit(int(biased_wt[i][p+7][j][k]),int(biased_wt[i][p+6][j][k]))
                                str_wt += str(w0w1) + ', ' + str(w2w3)+ ', ' + str(w4w5) + ', ' + str(w6w7) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt

            elif act_input_bits == 2:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(0, m[2*(lay-1)].weight.data.size(1), 8):
                                w0w1w2w3 = concat2bit(int(biased_wt[i][p+3][j][k]),int(biased_wt[i][p+2][j][k]),int(biased_wt[i][p+1][j][k]),int(biased_wt[i][p][j][k]))
                                w8w9w10w11 = concat2bit(int(biased_wt[i][p+11][j][k]),int(biased_wt[i][p+10][j][k]),int(biased_wt[i][p+9][j][k]),int(biased_wt[i][p+8][j][k]))
                                w4w5w6w7 = concat2bit(int(biased_wt[i][p+7][j][k]),int(biased_wt[i][p+6][j][k]),int(biased_wt[i][p+5][j][k]),int(biased_wt[i][p+4][j][k]))
                                w12w13w14w15 = concat2bit(int(biased_wt[i][p+15][j][k]),int(biased_wt[i][p+14][j][k]),int(biased_wt[i][p+13][j][k]),int(biased_wt[i][p+12][j][k]))
                                str_wt += str(w0w1w2w3) + ', ' + str(w8w9w10w11)+ ', ' + str(w4w5w6w7) + ', ' + str(w12w13w14w15) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt
                                
        else: #depthwise
            for i in range(m[2*(lay-1)].weight.size(1)):
                for j in range(m[2*(lay-1)].weight.data.size(2)):
                    for k in range(m[2*(lay-1)].weight.data.size(2)):
                        for p in range(0,m[2*(lay-1)].weight.data.size(0),2):
                            str_wt += str(concat4bit(int(biased_wt[p+1][i][j][k]),int(biased_wt[p][i][j][k]))) + ', ' 
                        str_wt += '\\\n'
            str_wt = str_wt[:-4]+'}'
            return str_wt
        
        
def generate_weights_PL_wt2bit(m,lay,act_input_bits,param_list):
    str_wt = '#define CONV'+ str(lay)+ '_WT {\\\n' 
    wt_bits = param_list[lay-1]['w_bits']
    Z_wt = - m[2*(lay-1)].weight.data.min() 
    biased_wt = (m[2*(lay-1)].weight.data + Z_wt)
    if wt_bits == 2:
        if param_list[lay-1]['quant_conv'].groups == 1: #convolve
            if act_input_bits == 8:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(0, m[2*(lay-1)].weight.data.size(1)-15, 16):
                                w0w1w4w5 = concat2bit(int(biased_wt[i][p+5][j][k]),int(biased_wt[i][p+4][j][k]),int(biased_wt[i][p+1][j][k]),int(biased_wt[i][p][j][k]))
                                w8w9w12w13 = concat2bit(int(biased_wt[i][p+13][j][k]),int(biased_wt[i][p+12][j][k]),int(biased_wt[i][p+9][j][k]),int(biased_wt[i][p+8][j][k]))
                                w2w3w6w7 = concat2bit(int(biased_wt[i][p+7][j][k]),int(biased_wt[i][p+6][j][k]),int(biased_wt[i][p+3][j][k]),int(biased_wt[i][p+2][j][k]))
                                w10w11w14w15 = concat2bit(int(biased_wt[i][p+15][j][k]),int(biased_wt[i][p+14][j][k]),int(biased_wt[i][p+11][j][k]),int(biased_wt[i][p+10][j][k]))
                                str_wt += str(w0w1w4w5) + ', ' + str(w8w9w12w13)+ ', ' + str(w2w3w6w7) + ', ' + str(w10w11w14w15) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt

            elif act_input_bits == 4:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(0, m[2*(lay-1)].weight.data.size(1)-15, 16):
                                w0w1w2w3 = concat2bit(int(biased_wt[i][p+3][j][k]),int(biased_wt[i][p+2][j][k]),int(biased_wt[i][p+1][j][k]),int(biased_wt[i][p][j][k]))
                                w8w9w10w11 = concat2bit(int(biased_wt[i][p+11][j][k]),int(biased_wt[i][p+10][j][k]),int(biased_wt[i][p+9][j][k]),int(biased_wt[i][p+8][j][k]))
                                w4w5w6w7 = concat2bit(int(biased_wt[i][p+7][j][k]),int(biased_wt[i][p+6][j][k]),int(biased_wt[i][p+5][j][k] ),int(biased_wt[i][p+4][j][k]))
                                w12w13w14w15 = concat2bit(int(biased_wt[i][p+15][j][k]),int(biased_wt[i][p+14][j][k]),int(biased_wt[i][p+13][j][k]),int(biased_wt[i][p+12][j][k]))
                                str_wt += str(w0w1w2w3) + ', ' + str(w8w9w10w11)+ ', ' + str(w4w5w6w7) + ', ' + str(w12w13w14w15) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt
            
            elif act_input_bits == 2:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(0, m[2*(lay-1)].weight.data.size(1)-15, 16):
                                w0w1w2w3 = concat2bit(int(biased_wt[i][p+3][j][k]),int(biased_wt[i][p+2][j][k]),int(biased_wt[i][p+1][j][k]),int(biased_wt[i][p][j][k]))
                                w8w9w10w11 = concat2bit(int(biased_wt[i][p+11][j][k]),int(biased_wt[i][p+10][j][k]),int(biased_wt[i][p+9][j][k]),int(biased_wt[i][p+8][j][k]))
                                w4w5w6w7 = concat2bit(int(biased_wt[i][p+7][j][k]),int(biased_wt[i][p+6][j][k]),int(biased_wt[i][p+5][j][k] ),int(biased_wt[i][p+4][j][k]))
                                w12w13w14w15 = concat2bit(int(biased_wt[i][p+15][j][k]),int(biased_wt[i][p+14][j][k]),int(biased_wt[i][p+13][j][k]),int(biased_wt[i][p+12][j][k]))
                                str_wt += str(w0w1w2w3) + ', ' + str(w4w5w6w7)+ ', ' + str(w8w9w10w11) + ', ' + str(w12w13w14w15) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt

        else: #depthwise
            for i in range(m[2*(lay-1)].weight.size(1)):
                for j in range(m[2*(lay-1)].weight.data.size(2)):
                    for k in range(m[2*(lay-1)].weight.data.size(2)):
                        for p in range(0,m[2*(lay-1)].weight.data.size(0),4):
                            str_wt += str(concat2bit(int(biased_wt[p+3][i][j][k]),int(biased_wt[p+2][i][j][k]),int(biased_wt[p+1][i][j][k]),int(biased_wt[p][i][j][k]))) + ', ' 
                        str_wt += '\\\n'
            str_wt = str_wt[:-4]+'}'
            return str_wt
            
def generate_weights_PC_wt8bit(m,lay,act_input_bits,param_list):
    str_wt = '#define CONV'+ str(lay)+ '_WT {\\\n' 
    wt_bits = param_list[lay-1]['w_bits']
    Z_wt = torch.Tensor(m[2*(lay-1)].weight.data.size(0))
    biased_wt = torch.Tensor(m[2*(lay-1)].weight.data.size())
    for i in range(m[2*(lay-1)].weight.data.size(0)):
        Z_wt[i] = - m[2*(lay-1)].weight.data[i].min() 
        biased_wt[i] = m[2*(lay-1)].weight.data[i] + Z_wt[i]
    if wt_bits == 8: #order of weights depends on the bitwise
        if param_list[lay-1]['quant_conv'].groups == 1: #order is different if the layer function is depthwise (groups!=1) or convolve (odd layers)
            if act_input_bits == 8:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    biased_wt[i] = (m[2*(lay-1)].weight.data[i] + Z_wt[i])
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(m[2*(lay-1)].weight.data.size(1)):
                                str_wt += str(int(biased_wt[i][p][j][k])) + ', '
                            if lay == 1: #the first layer has to be zero padded
                                str_wt += str(int(Z_wt[i]) if Z_wt[i]<255 else 255) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt

            elif act_input_bits == 4:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    biased_wt[i] = (m[2*(lay-1)].weight.data[i] + Z_wt[i])
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(0, m[2*(lay-1)].weight.data.size(1), 8):
                                str_wt += str(int(biased_wt[i][p][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+1][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+4][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+5][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+2][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+3][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+6][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+7][j][k])) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt

            elif act_input_bits == 2:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    biased_wt[i] = (m[2*(lay-1)].weight.data[i] + Z_wt[i])
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(0, m[2*(lay-1)].weight.data.size(1), 16):
                                str_wt += str(int(biased_wt[i][p][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+1][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+8][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+9][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+2][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+3][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+10][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+11][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+4][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+5][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+12][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+13][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+6][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+7][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+14][j][k])) + ', '
                                str_wt += str(int(biased_wt[i][p+15][j][k])) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt
                
        else:  #depthwise
            for i in range(m[2*(lay-1)].weight.size(1)):
                biased_wt[i] = (m[2*(lay-1)].weight.data[i] + Z_wt[i])
                for j in range(m[2*(lay-1)].weight.data.size(2)):
                    for k in range(m[2*(lay-1)].weight.data.size(2)):
                        for p in range(m[2*(lay-1)].weight.data.size(0)):
                            str_wt += str(int(biased_wt[p][i][j][k])) + ', '
                        str_wt += '\\\n'
            str_wt = str_wt[:-4]+'}'
            return str_wt
                                              
            

def generate_weights_PC_wt4bit(m,lay,act_input_bits,param_list):
    str_wt = '#define CONV'+ str(lay)+ '_WT {\\\n' 
    wt_bits = param_list[lay-1]['w_bits']
    Z_wt = torch.Tensor(m[2*(lay-1)].weight.data.size(0))
    biased_wt = torch.Tensor(m[2*(lay-1)].weight.data.size())
    for i in range(m[2*(lay-1)].weight.data.size(0)):
        Z_wt[i] = - m[2*(lay-1)].weight.data[i].min() 
        biased_wt[i] = m[2*(lay-1)].weight.data[i] + Z_wt[i]
    if wt_bits == 4: 
        if param_list[lay-1]['quant_conv'].groups == 1: #convolve    
            if act_input_bits == 8:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    biased_wt[i] = (m[2*(lay-1)].weight.data[i] + Z_wt[i])
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(0, m[2*(lay-1)].weight.data.size(1)-7, 8):
                                w0w1 = concat4bit(int(biased_wt[i][p+1][j][k]),int(biased_wt[i][p][j][k]))
                                w4w5 = concat4bit(int(biased_wt[i][p+5][j][k]),int(biased_wt[i][p+4][j][k]))
                                w2w3 = concat4bit(int(biased_wt[i][p+3][j][k]),int(biased_wt[i][p+2][j][k]))
                                w6w7 = concat4bit(int(biased_wt[i][p+7][j][k]),int(biased_wt[i][p+6][j][k]))
                                str_wt += str(w0w1) + ', ' + str(w4w5)+ ', ' + str(w2w3) + ', ' + str(w6w7) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt
                       
            elif act_input_bits == 4:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    biased_wt[i] = (m[2*(lay-1)].weight.data[i] + Z_wt[i])
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(0, m[2*(lay-1)].weight.data.size(1)-7, 8):
                                w0w1 = concat4bit(int(biased_wt[i][p+1][j][k]),int(biased_wt[i][p][j][k]))
                                w2w3 = concat4bit(int(biased_wt[i][p+3][j][k]),int(biased_wt[i][p+2][j][k]))
                                w4w5 = concat4bit(int(biased_wt[i][p+5][j][k]),int(biased_wt[i][p+4][j][k]))
                                w6w7 = concat4bit(int(biased_wt[i][p+7][j][k]),int(biased_wt[i][p+6][j][k]))
                                str_wt += str(w0w1) + ', ' + str(w2w3)+ ', ' + str(w4w5) + ', ' + str(w6w7) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt
         
            elif act_input_bits == 2:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    biased_wt[i] = (m[2*(lay-1)].weight.data[i] + Z_wt[i])
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(0, m[2*(lay-1)].weight.data.size(1), 8):
                                w0w1w2w3 = concat2bit(int(biased_wt[i][p+3][j][k]),int(biased_wt[i][p+2][j][k]),int(biased_wt[i][p+1][j][k]),int(biased_wt[i][p][j][k]))
                                w8w9w10w11 = concat2bit(int(biased_wt[i][p+11][j][k]),int(biased_wt[i][p+10][j][k]),int(biased_wt[i][p+9][j][k]),int(biased_wt[i][p+8][j][k]))
                                w4w5w6w7 = concat2bit(int(biased_wt[i][p+7][j][k]),int(biased_wt[i][p+6][j][k]),int(biased_wt[i][p+5][j][k]),int(biased_wt[i][p+4][j][k]))
                                w12w13w14w15 = concat2bit(int(biased_wt[i][p+15][j][k]),int(biased_wt[i][p+14][j][k]),int(biased_wt[i][p+13][j][k]),int(biased_wt[i][p+12][j][k]))
                                str_wt += str(w0w1w2w3) + ', ' + str(w8w9w10w11)+ ', ' + str(w4w5w6w7) + ', ' + str(w12w13w14w15) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt
                              
        else: #depthwise
            for i in range(m[2*(lay-1)].weight.size(1)):
                biased_wt[i] = (m[2*(lay-1)].weight.data[i] + Z_wt[i])
                for j in range(m[2*(lay-1)].weight.data.size(2)):
                    for k in range(m[2*(lay-1)].weight.data.size(2)):
                        for p in range(0,m[2*(lay-1)].weight.data.size(0),2):
                            str_wt += str(concat4bit(int(biased_wt[p+1][i][j][k]),int(biased_wt[p][i][j][k]))) + ', ' 
                        str_wt += '\\\n'
            str_wt = str_wt[:-4]+'}'
            return str_wt
        

def generate_weights_PC_wt2bit(m,lay,act_input_bits,param_list):
    str_wt = '#define CONV'+ str(lay)+ '_WT {\\\n' 
    wt_bits = param_list[lay-1]['w_bits']
    Z_wt = torch.Tensor(m[2*(lay-1)].weight.data.size(0))
    biased_wt = torch.Tensor(m[2*(lay-1)].weight.data.size())
    for i in range(m[2*(lay-1)].weight.data.size(0)):
        Z_wt[i] = - m[2*(lay-1)].weight.data[i].min() 
        biased_wt[i] = m[2*(lay-1)].weight.data[i] + Z_wt[i]
    if wt_bits == 2:
        if param_list[lay-1]['quant_conv'].groups == 1: #convolve
            if act_input_bits == 8:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    biased_wt[i] = (m[2*(lay-1)].weight.data[i] + Z_wt[i])
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(0, m[2*(lay-1)].weight.data.size(1)-15, 16):
                                w0w1w4w5 = concat2bit(int(biased_wt[i][p+5][j][k]),int(biased_wt[i][p+4][j][k]),int(biased_wt[i][p+1][j][k]),int(biased_wt[i][p][j][k]))
                                w8w9w12w13 = concat2bit(int(biased_wt[i][p+13][j][k]),int(biased_wt[i][p+12][j][k]),int(biased_wt[i][p+9][j][k]),int(biased_wt[i][p+8][j][k]))
                                w2w3w6w7 = concat2bit(int(biased_wt[i][p+7][j][k]),int(biased_wt[i][p+6][j][k]),int(biased_wt[i][p+3][j][k]),int(biased_wt[i][p+2][j][k]))
                                w10w11w14w15 = concat2bit(int(biased_wt[i][p+15][j][k]),int(biased_wt[i][p+14][j][k]),int(biased_wt[i][p+11][j][k]),int(biased_wt[i][p+10][j][k]))
                                str_wt += str(w0w1w4w5) + ', ' + str(w8w9w12w13)+ ', ' + str(w2w3w6w7) + ', ' + str(w10w11w14w15) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt
                                
            elif act_input_bits == 4:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    biased_wt[i] = (m[2*(lay-1)].weight.data[i] + Z_wt[i])
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(0, m[2*(lay-1)].weight.data.size(1)-15, 16):
                                w0w1w2w3 = concat2bit(int(biased_wt[i][p+3][j][k]),int(biased_wt[i][p+2][j][k]),int(biased_wt[i][p+1][j][k]),int(biased_wt[i][p][j][k]))
                                w8w9w10w11 = concat2bit(int(biased_wt[i][p+11][j][k]),int(biased_wt[i][p+10][j][k]),int(biased_wt[i][p+9][j][k]),int(biased_wt[i][p+8][j][k]))
                                w4w5w6w7 = concat2bit(int(biased_wt[i][p+7][j][k]),int(biased_wt[i][p+6][j][k]),int(biased_wt[i][p+5][j][k] ),int(biased_wt[i][p+4][j][k]))
                                w12w13w14w15 = concat2bit(int(biased_wt[i][p+15][j][k]),int(biased_wt[i][p+14][j][k]),int(biased_wt[i][p+13][j][k]),int(biased_wt[i][p+12][j][k]))
                                str_wt += str(w0w1w2w3) + ', ' + str(w8w9w10w11)+ ', ' + str(w4w5w6w7) + ', ' + str(w12w13w14w15) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt

            elif act_input_bits == 2:
                for i in range(m[2*(lay-1)].weight.data.size(0)):
                    biased_wt[i] = (m[2*(lay-1)].weight.data[i] + Z_wt[i])
                    for j in range(m[2*(lay-1)].weight.data.size(2)):
                        for k in range(m[2*(lay-1)].weight.data.size(2)):
                            for p in range(0, m[2*(lay-1)].weight.data.size(1)-15, 16):
                                w0w1w2w3 = concat2bit(int(biased_wt[i][p+3][j][k]),int(biased_wt[i][p+2][j][k]),int(biased_wt[i][p+1][j][k]),int(biased_wt[i][p][j][k]))
                                w8w9w10w11 = concat2bit(int(biased_wt[i][p+11][j][k]),int(biased_wt[i][p+10][j][k]),int(biased_wt[i][p+9][j][k]),int(biased_wt[i][p+8][j][k]))
                                w4w5w6w7 = concat2bit(int(biased_wt[i][p+7][j][k]),int(biased_wt[i][p+6][j][k]),int(biased_wt[i][p+5][j][k] ),int(biased_wt[i][p+4][j][k]))
                                w12w13w14w15 = concat2bit(int(biased_wt[i][p+15][j][k]),int(biased_wt[i][p+14][j][k]),int(biased_wt[i][p+13][j][k]),int(biased_wt[i][p+12][j][k]))
                                str_wt += str(w0w1w2w3) + ', ' + str(w4w5w6w7)+ ', ' + str(w8w9w10w11) + ', ' + str(w12w13w14w15) + ', '
                    str_wt += '\\\n'
                str_wt = str_wt[:-4]+'}'
                return str_wt

        else: #depthwise
            for i in range(m[2*(lay-1)].weight.size(1)):
                biased_wt[i] = (m[2*(lay-1)].weight.data[i] + Z_wt[i])
                for j in range(m[2*(lay-1)].weight.data.size(2)):
                    for k in range(m[2*(lay-1)].weight.data.size(2)):
                        for p in range(0,m[2*(lay-1)].weight.data.size(0),4):
                            str_wt += str(concat2bit(int(biased_wt[p+3][i][j][k]),int(biased_wt[p+2][i][j][k]),int(biased_wt[p+1][i][j][k]),int(biased_wt[p][i][j][k]))) + ', ' 
                        str_wt += '\\\n'
            str_wt = str_wt[:-4]+'}'
            return str_wt
    
def compute_padding_same( IM_DIM, KER_DIM, STRIDE, DILATION ):
    input_rows = IM_DIM
    filter_rows = KER_DIM
    effective_filter_size_rows = (filter_rows - 1) * DILATION + 1
    out_rows = (input_rows + STRIDE - 1) // STRIDE
    padding_needed = max(0, (out_rows - 1) * STRIDE + effective_filter_size_rows -
                  input_rows)
    padding_rows = max(0, (out_rows - 1) * STRIDE +
                        (filter_rows - 1) * DILATION + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * STRIDE +
                        (filter_rows - 1) * DILATION + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    print(padding_cols//2, padding_rows//2)
    if rows_odd or cols_odd:
        return [int(padding_cols//2), int(padding_cols//2)+int(cols_odd), int(padding_rows//2), int(padding_rows//2)+int(rows_odd)]
    else:
        return [int(padding_cols//2), int(padding_cols//2), int(padding_rows//2), int(padding_rows//2) ]

#support functions
def concat4bit(a,b): 
    return a*(2**4)+b

def concat2bit(a,b,c,d):
    return a*(2**6)+b*(2**4)+c*(2**2)+d