import mxnet as mx
import torch as th
import numpy as np

#params = mx.nd.load('checkpoints/checkpoints/shuai/ckpt_stage2_lamb_32k_sz/0001564.params')
#params = mx.nd.load('checkpoints/checkpoints/32-nodes/debug-tf-hd5-fp32-again/0001563.params')
#params = mx.nd.load('checkpoints/checkpoints/shuai/ckpt_stage2_lamb_32k_mean_sz/0001564.params')
#params = mx.nd.load('checkpoints/checkpoints/32-nodes/debug-tf-hd5-fp32-64k-bound/0001563.params')
#params = mx.nd.load('checkpoints/checkpoints/shuai/ckpt_stage2_lamb_32k_mean_sz/0001564-2.params')
#params = mx.nd.load('checkpoints/checkpoints/32-nodes/debug-tf-hd5-v2-fp32-64k-005-0444/0001499.params')
#params = mx.nd.load('checkpoints/checkpoints/32-nodes/debug-tf-hd5-v2-fp32-64k-005-0444/0000999.params')
#params = mx.nd.load('checkpoints/checkpoints/32-nodes/debug-tf-hd5-v2-fp16-64k-004/0001563.params')
#params = mx.nd.load('checkpoints/checkpoints/32-nodes/debug-tf-hd5-v2-fp16-64k-phase2-004-03/0001563.params')
#params = mx.nd.load('checkpoints/checkpoints/32-nodes/debug-tf-hd5-v2-fp16-64k-phase2-004-04/0001563.params')
#params = mx.nd.load('checkpoints/checkpoints/32-nodes/debug-tf-hd5-v2-fp16-64k-phase2-002-04/0001563.params')
#params = mx.nd.load('checkpoints/checkpoints/64-nodes/tf-64-node-fp32-32k/0001564.params')
#params = mx.nd.load('checkpoints/checkpoints/64-nodes/tf-64-node-fp16-64k-no-cudnn-0.3/0001564.params')
#params = mx.nd.load('checkpoints/checkpoints/64-nodes/tf-64-node-fp16-64k-no-cudnn-0.3/0000782.params')
#params = mx.nd.load('checkpoints/checkpoints/tf-64-node-fp16-64k-with-cudnn/0001564.params')
#params = mx.nd.load('checkpoints/checkpoints/tf-64-node-fp16-64k-sm-len/0001564.params')
#params = mx.nd.load('checkpoints/checkpoints/tf-64-node-fp16-64k-speed-gelu-sm/0001564.params')
#params = mx.nd.load('checkpoints/checkpoints/tf-64-node-fp16-64k-speed-gelu/0001564.params')
#params = mx.nd.load('checkpoints/checkpoints/container-64k-512-ckpt/0007038.params')
#params = mx.nd.load('checkpoints/checkpoints/64-nodes/container-32k-512-no-gelu-base-full-len-ckpt/haibinlin/bert-docker:test/0015640.params')
#params = mx.nd.load('checkpoints/checkpoints/64-nodes/container-128k-512-no-gelu-ckpt/haibinlin/bert-docker-test/0001564.params')
#params = mx.nd.load('checkpoints/checkpoints/shuai/debug/0001563.params')
#params = mx.nd.load('checkpoints/checkpoints/64-nodes/container-96k-512-ckpt/haibinlin/bert-docker:eb4930ce/0001564.params')
#params = mx.nd.load('checkpoints/checkpoints/64-nodes/container-96k-512-whole-ckpt/haibinlin/bert-docker:29f85a5f/0001564.params')
params = mx.nd.load('checkpoints/checkpoints/64-nodes/container-96k-512-ckpt/haibinlin/bert-docker-29f85a5f/0001564.params')

mapping = {
'encoder.transformer_cells': 'bert.encoder.layer',
'attention_cell.proj_': 'attention.self.',
'proj': 'attention.output.dense',
'ffn.ffn_1': 'intermediate.dense_act',
'ffn.ffn_2': 'output.dense',
'ffn.layer_norm': 'output.LayerNorm',
'pooler':'bert.pooler.dense_act',
'decoder.0':'cls.predictions.transform.dense_act',
'decoder.2':'cls.predictions.transform.LayerNorm',
'decoder.3.bias':'cls.predictions.bias',
'decoder.3.weight':'bert.embeddings.word_embeddings.weight',
'classifier':'cls.seq_relationship',
'gamma':'weight',
'beta':'bias',
'encoder.layer_norm':'bert.embeddings.LayerNorm',
'token_type_embed.0.weight':'bert.embeddings.token_type_embeddings.weight',
'word_embed.0.weight':'bert.embeddings.word_embeddings.weight',
'encoder.position_weight':'bert.embeddings.position_embeddings.weight',
}
secondary_map = {'layer_norm': 'attention.output.LayerNorm',}

# set parameter data
pt_params = {}
pt_params['model'] = {}
for name in params:
    pytorch_name = name
    for k, v in mapping.items():
        pytorch_name = pytorch_name.replace(k, v)
    for k, v in secondary_map.items():
        pytorch_name = pytorch_name.replace(k, v)
    mx_array = params[name]
    if (mx_array.shape == (30522, 1024) or mx_array.shape == (30522, 768)):
        import gluonnlp as nlp
        _, vocab = nlp.model.get_model('bert_24_1024_16', dataset_name='book_corpus_wiki_en_uncased', pretrained=False)
        np_arry = np.zeros((30528, mx_array.shape[1]))
        mx_array = mx_array.asnumpy()
        # convert based on the vocab
        if False:
            print('convert mx array')
            with open('data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt', 'r') as f:
                idx = 0
                for line in f:
                    word = line[:-1]
                    assert word in vocab
                    np_arry[idx] = mx_array[vocab[word]]
                    idx += 1
                    print('covnerted ', idx)
        else:
            np_arry[:30522] = mx_array
            print('not converting mx array')
    else:
        np_arry = mx_array.astype('float32').asnumpy()
    assert np.isfinite(np.linalg.norm(np_arry))
    pt_params['model'][pytorch_name] = th.Tensor(np_arry)

#th.save(pt_params, 'checkpoints/checkpoints/shuai/ckpt_stage2_lamb_32k_sz/0001564.pt')
#th.save(pt_params, 'checkpoints/checkpoints/32-nodes/debug-tf-hd5-fp32-again/0001563.pt')
#th.save(pt_params, 'checkpoints/checkpoints/shuai/ckpt_stage2_lamb_32k_mean_sz/0001564.pt')
#th.save(pt_params, 'checkpoints/checkpoints/32-nodes/debug-tf-hd5-fp32-64k-bound/0001563.pt')
#th.save(pt_params, 'checkpoints/checkpoints/32-nodes/debug-tf-hd5-fp32-64k-bound/0001563-2.pt')
#th.save(pt_params, 'checkpoints/checkpoints/shuai/ckpt_stage2_lamb_32k_mean_sz/0001564-2.pt')
#th.save(pt_params, 'checkpoints/checkpoints/32-nodes/debug-tf-hd5-v2-fp32-64k-005-0444/0001563.params')
#th.save(pt_params, 'checkpoints/checkpoints/32-nodes/debug-tf-hd5-v2-fp32-64k-005-0444/0000999.pt')
#th.save(pt_params, 'checkpoints/checkpoints/32-nodes/debug-tf-hd5-v2-fp16-64k-phase2-004-03/0001563.pt')
#th.save(pt_params, 'checkpoints/checkpoints/32-nodes/debug-tf-hd5-v2-fp16-64k-phase2-004-04/0001563.pt')
#th.save(pt_params, 'checkpoints/checkpoints/32-nodes/debug-tf-hd5-v2-fp16-64k-phase2-002-04/0001563.pt')
#th.save(pt_params, 'checkpoints/checkpoints/')
#th.save(pt_params, 'checkpoints/checkpoints/64-nodes/tf-64-node-fp32-32k/0001564.pt')
#th.save(pt_params, 'checkpoints/checkpoints/64-nodes/tf-64-node-fp16-64k-no-cudnn-0.3/0001564.pt')
#th.save(pt_params, 'checkpoints/checkpoints/tf-64-node-fp16-64k-with-cudnn/0001564.pt')
#th.save(pt_params, 'checkpoints/checkpoints/tf-64-node-fp16-64k-sm-len/0001564.pt')
#th.save(pt_params, 'checkpoints/checkpoints/tf-64-node-fp16-64k-speed-gelu-sm/0001564.pt')
#th.save(pt_params, 'checkpoints/checkpoints/tf-64-node-fp16-64k-speed-gelu/0001564.pt')
#th.save(pt_params, 'checkpoints/checkpoints/container-64k-512-ckpt/0007038.pt')
#th.save(pt_params, 'checkpoints/checkpoints/64-nodes/container-32k-512-no-gelu-base-full-len-ckpt/haibinlin/bert-docker:test/0015640.pt')
#th.save(pt_params, 'checkpoints/checkpoints/64-nodes/container-128k-512-no-gelu-ckpt/haibinlin/bert-docker-test/0001564.pt')
#th.save(pt_params, 'checkpoints/checkpoints/shuai/debug/0001563.pt')
#th.save(pt_params, 'checkpoints/checkpoints/64-nodes/container-96k-512-ckpt/haibinlin/bert-docker:eb4930ce/0001564.pt')
#th.save(pt_params, 'checkpoints/checkpoints/64-nodes/container-96k-512-whole-ckpt/haibinlin/bert-docker:29f85a5f/0001564.pt')
th.save(pt_params, 'checkpoints/checkpoints/64-nodes/container-96k-512-ckpt/haibinlin/bert-docker-29f85a5f/0001564.pt')
