with open('input.txt','r', encoding='utf-8') as f:
  text=f.read()# a string

import torch
import torch.nn as nn
from torch.nn import functional as f
torch.manual_seed(1337)
#initialization
block_size=256 #max context size
batch_size=64
num_iter=5000
learning_rate=3e-4
eval_iter=300
eval_frequency=500
n_embed=384
dropout=0.2#probability of a neuron to be shut off which is done randomly in each forward pass
n_head=6
device='cuda' if torch.cuda.is_available() else "cpu"

#prepare vocabulary and encode/decode(ie tokenization)
vocab=sorted(list(set(text)))
vocab_size=len(vocab) #=65
stoi={s:i for i,s in enumerate(vocab)}
itos={i:s for i,s in enumerate(vocab)}
encoder=lambda x:[stoi[c] for c in x]
decoder=lambda x:"".join([itos[i] for i in x])

#train/val separation
integers=torch.tensor(encoder(text),dtype=torch.long) #later down the line the nn.Embedding layer expects input to be of dtype=longint
n=int(0.9*len(integers))
train_set=integers[:n]
val_set=integers[n:]

# x=train_set[:block_size]
# y=train_set[1:block_size+1]
# for i in range(block_size):
#   context=x[:i+1]
#   target=y[i]
#   print(f"context:{context} target:{target}")
# The main concept here is that the transformer must be able to take inputs that are of different sizes specifically between size of (1-block_size)

#prepare dataset
def data_prep(dat_type):
  data=train_set if dat_type=="train" else val_set
  ix=torch.randint(len(data)-block_size,(batch_size,)) #generates random tensors of size=batch_size
  #we subtract data with block _size here because ix[i] (where i is any index ) must be such that the data contains number of
  #elements equal to block_size after ix[i] so that we dont accidently access tensor having index out of range in the step below
  #for example if len(data)=500 then if ix[9]=496 then in step below we try to access x=data[496:504]which is out of range
  x=torch.stack([data[i:i+block_size] for i in ix] )
  y=torch.stack([data[i+1:i+block_size+1] for i in ix]) #for a input sequence like "sayujp" the target contains the outputs for all the possible context size like x="sayuj" then y="ayujp" where for context=s output=a for context=sa output=y ... for context=sayuj output=p
  #This removes the need to train for different context length separately 
  x,y=x.to(device),y.to(device)#(batch_size,block_size)
  return x,y
#calculate the estimate loss 
#we calculate the average loss across all batches per each iteration of training to find an accurate estimate of losses since the loss across batches
#tends to have oscillating nature
with torch.no_grad():
  def estimate_loss():
    out={}
    model.eval() #model is set to evaluation mode to turn of dropout and use running mean and std in batchnorm to achieve consistent result during inference 
    for split in ['train','val']:
      losses=torch.zeros(eval_iter)
      for i in range(eval_iter):
        xb,yb=data_prep(split)
        logits,loss=model(xb,yb)
        losses[i]=loss.item()
      out[split]=losses.mean()
    model.train()#model is set to train mode so that remaining tasks like backward pass,updating weights can be carried out in training mode
    return out

class LayersNorm1d:
  def __init__(self,num_features,eps=1e-05):
    self.bngain=torch.ones(num_features)
    self.bnbias=torch.zeros(num_features)
    self.eps=eps
  def __call__(self,x):
    bmean=x.mean(1,keepdims=True)# here in (B,T,n_embed) both B,T act as a batch dimension and thus 1 here means the feature dimension of each token independently
    bvar=x.var(1,keepdims=True)
    self.out=self.bngain*((x-bmean)/torch.sqrt(bvar+self.eps))+self.bnbias
    return self.out
  def parameters(self):
    return [self.bngain,self.bnbias]


#single self-attention Head
class Head(nn.Module):
  def __init__(self,head_size):
    super().__init__()
    self.key=nn.Linear(n_embed,head_size,bias=False)
    self.query=nn.Linear(n_embed,head_size,bias=False)
    self.value=nn.Linear(n_embed,head_size,bias=False) 
    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size))) 
    self.drop_out=nn.Dropout(dropout)
  def forward(self,x):
    B,T,C=x.shape
    k=self.key(x) #what this token has to offer to the other token #(B,T,head_size)
    v=self.value(x) #the information about a token (that is obtained after training it on various datasets) that is passed between tokens #(B,T,head_size) 
    #for example a token "bank" may have many meaning depending on the context so various meaning of this token is learned and packed as a single vector v
    q=self.query(x) #what this token is looking for in another token #(B,T,head_size)
    wei=q@k.transpose(-1,-2)*k.shape[-1]**-0.5#(B,T,T) #This is the attention matrix which gives nos that specify how strongly this questions and answers(i.e query and keys) align with each other
    #masking of attention matrix:
    wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))#ensuring info from future tokens can't pass to the past tokens#
    wei=f.softmax(wei,-1)#We normalize across the last dimension because for a given token,we calculate the probabilities of keys that correspondes to high attention scores
    #so in essence we are calculating the conditional probability that given a query what are the probability distribution of it aligning with keys of all the tokens in the given block
    wei=self.drop_out(wei)
    out=wei@v#here value vector is projected into smaller dimension so that in each head value vector can learn different features and meanings of a single word
    #we're taking a weighted sum of the value vectors across tokens, where each weight comes from the probability distribution given by the attention matrix (derived from the query-key interactions). Each token's output vector is effectively a blend of value vectors from other tokens, weighted according to how relevant (or similar) those tokens are, as determined by the attention scores.
    return out#(B,T,T) @(B,T,head_size)=(B,T,head_size) 

#Multi-Head attention:
class MultiHead(nn.Module):
  def __init__(self,numhead,head_size):
    super().__init__()
    self.heads=nn.ModuleList([Head(head_size) for _ in range(numhead)])
    self.proj=nn.Linear(n_embed,n_embed)
    self.drop_out=nn.Dropout(dropout)
  def forward(self,x):
    out=torch.cat([h(x) for h in self.heads],dim=-1) #(B,T,n_embed)=(B,T,n_embed//4)+(B,T,n_embed//4)+(B,T,n_embed//4)+(B,T,n_embed//4)
    out=self.drop_out(self.proj(out))
    return out 

#FeedForward layer:
class FeedForward(nn.Module):
  def __init__(self):
    super().__init__()
    self.net=nn.Sequential(
      nn.Linear(n_embed,4*n_embed), #Temporarily expand the feature space by times 4 for richer learning.
      nn.ReLU(),
      nn.Linear(4*n_embed,n_embed), #project back to original n_embed size
      nn.Dropout(dropout)
    )
  def forward(self,x):
    out=self.net(x)
    return out 

#Blocks
class Block(nn.Module):
  def __init__(self,n_head):
    super().__init__()
    head_size=n_embed//n_head
    self.attention=MultiHead(n_head,head_size)   
    self.feedforward=FeedForward()
    self.ln1=nn.LayerNorm(n_embed)
    self.ln2=nn.LayerNorm(n_embed) 
  def forward(self,x):
    x=x+self.attention(self.ln1(x))
    x=x+self.feedforward(self.ln2(x))
    return x

#BigramModel footprint
class BigramModel(nn.Module):#inheritance from nn.Module: 1>keeps track of trainable parameters 2>provide builtin function like .train(),.eval() 3> when we initilize the object and call like out=m(Xb,Yb) the forward function is automatically executed
  def __init__(self,):
    super().__init__()#This is required to initialize the parent class's internal state and parameters
    self.embedding=nn.Embedding(vocab_size,n_embed)    #This creates an embeding table/matrix of (vocab_size,no_of_dimension)
    self.positional_embedding=nn.Embedding(block_size,n_embed) #These encodings are learned by the transformers itself but limits the input sequence to be fixed of size=block_length
    self.linear_block=nn.Linear(n_embed,vocab_size)
    self.blocks=nn.Sequential(
      Block(n_head=n_head),
      Block(n_head=n_head),
      Block(n_head=n_head),
      nn.LayerNorm(n_embed)
    )
  def forward(self,x,target=None):#similar to that of __call__ called when an object of BigramModel receives an input argument
    B,T=x.shape#any block size may come as input so use T not block_size
    tok_emb=self.embedding(x)#each element of x plucks out a row of embedding matrix #(B,T,n_embed)
    pos_emb=self.positional_embedding(torch.arange(T,device=device))#(T,n_embed)
    total_emb=pos_emb+tok_emb#pos_emb is broadcasted accross the batch
    b_out=self.blocks(total_emb)
    # attention=self.at_head(total_emb)#(B,T,n_embed)
    # ffd=self.feedforward(attention)#(B,T,n_embed)
    logits=self.linear_block(b_out)#(B,T,n_embed)@(n_embed,vocab_size)=(B,T,vocab_size)
    if target==None:
      loss=None
    else:
      B,T,C=logits.shape
      logits=logits.view(B*T,C) #B*T is done because it is a requirement in cross_entropy function below
      target=target.view(B*T)
      loss=f.cross_entropy(logits,target)
    return logits,loss
  def generate(self,x,maxtokens):
    for _ in range(maxtokens):#here x is of shape(B,T)
      xin=x[:,-block_size:] #we need to limit the context of batches to no more than last 8(i.e [-8:]) because pos_emb=self.positional_embedding(torch.arange(T),device=device) in BigramModel has only 8 max positions to encode
      logits,loss=self(xin)#(B,T,C)
      logits=logits[:,-1,:]#(B,C)
      probs=f.softmax(logits,dim=-1)#(B,C)
      ix=torch.multinomial(probs,num_samples=1)#(B,1) because multinomial treats one row as a separate probability distribution
      x=torch.cat((x,ix),dim=1)# here x's shape becomes (B,T+1)
    return x

#initialize
model=BigramModel()
m=model.to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)

#training loop
for i in range(num_iter):
  xb,yb=data_prep('train')
  out,loss=m(xb,yb)
  if i%eval_frequency==0:
    e_loss=estimate_loss()
    print(f"Training loss: {e_loss['train']}  Evaluation loss: {e_loss['val']} ")
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

print(decoder(model.generate(torch.zeros((1,1),dtype=torch.long,device=device),maxtokens=1000)[0].tolist()))




