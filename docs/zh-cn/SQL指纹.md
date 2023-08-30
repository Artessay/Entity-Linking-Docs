# SQL指纹

## SQL指纹提取

### SQL指纹概念

SQL指纹指将一条SQL中的字面值替换成其他固定符号。可以用来做SQL脱敏或者SQL归类。这在SQL应用中统计查询信息，分析热点查询模式中将发挥重要作用，有助于为查询优化指明前进方向。

SQL指纹的关键是**模型识别**和**信息脱敏**。在提取SQL指纹的过程中，我们需要识别SQL语句内出现的特定模式，并将其中的字面量进行替换。

传统上，SQL指纹的识别可以采用正则匹配SQL字符串或者将SQL转换成抽象语法树之后再进行解析等方式。完全通过字符串解析会使得代码及其复杂而难以阅读，好处是无需关心 SQL 语义。因此，从代码的可维护性以及可扩展性来看，采用抽象语法树的方式来解析SQL是更为合适的。

### SQL指纹提取实现

1. 通过parser将SQL解析成语法树
2. 修改语法树傻姑娘节点对应的值
3. 将语法树还原成SQL

## MD5

### MD5算法概念

MD5即Message-Digest Algorithm 5（信息-摘要算法5），用于确保信息传输完整一致。MD5算法是Hash算法的一种，叫做讯息摘要演算法。所谓摘要，从字面意思理解，是指内容的大概。在MD5算法中，这个摘要是指将任意数据映射成一个128位长的摘要信息。并且其是不可逆的，即从摘要信息无法反向推演中原文，在演算过程中，原文的内容也是有丢失的。

因为MD5算法最终生成的是一个128位长的数据，从原理上说，有2^128种可能，这是一个非常巨大的数据，约等于3.4乘10的38次方，虽然这个是个天文数字，但是世界上可以进行加密的数据原则上说是无限的，因此是可能存在不同的内容经过MD5加密后得到同样的摘要信息，但这个碰中的概率非常小。

### MD5使用场景

MD5常用在密码加密中，一般为了保证用户密码的安全，在数据库中存储的都是用户的密码经过MD5加密后的值，在客户端用户输入密码后，也会使用MD5进行加密，这样即使用户的网络被窃听，窃听者依然无法拿到用户的原始密码，并且即使用户数据库被盗，没有存储明文的密码对用户来说也多了一层安全保障。

MD5签名技术还常用于防止信息的篡改。使用MD5可以对信息进行签名，接收者拿到信息后只要重新计算签名和原始签名进行对比，即可知道数据信息是否中途被篡改了。

### MD5算法原理

MD5算法大致分为4步完成：

**第1步：进行数据填充整理**

    这一步是对要加密的数据进行填充和整理，将要加密的二进制数据对512取模，得到的结果如果不够448位，则进行补足，补足的方式是第1位填充1，后面全部填充0。

**第2步：记录数据长度**

    经过第一步整理完成后的数据的位数可以表示为N*512+448，再向其后追加以64位二进制表示的填充前信息长度（单位为Bit），如果二进制表示的填充前信息长度超过64位，则取低64位。比如数据的长度为16字节，则用10000来填充后64位。这一步做完后，数据的位数将变成(N+1)*512。

**第3步：以标准的幻数作为输入**

    MD5的实现需要每512个字节进行一次处理，后一次处理的输入为前一次处理的输出，因此，在循环处理开始之前，需要拿4个标准数作为输入。初始的128位值为初始链接变量，这些参数用于第一轮的运算，以大端字节序来表示，它们分别为： A=0x01234567，B=0x89ABCDEF，C=0xFEDCBA98，D=0x76543210。
    
    上面每一个变量给出的数值是高字节存于内存低地址，低字节存于内存高地址，即大端字节序。在程序中变量A、B、C、D的值分别是：

```c++
unsigned int A=0x67452301,B=0xefcdab89,C=0x98badcfe,D=0x10325476;
```

**第4步：进行N轮循环处理，将最后的结果输出**

    这一步重要的是每一轮的处理算法，每一轮处理也要循环64次，这64次循环被分为4组，每16次循环为一组，每组循环使用不同的逻辑处理函数，处理完成后，将输出作为输入进入下一轮循环。

    

### MD5算法实现

```c#
//将大端字节序转换为小端字节序
void convertToLittleEndian(unsigned int *data, int len);
//进行循环左移函数
void ROL(unsigned int *s, unsigned short cx);
//MD5加密函数
void MD5(NSString *str);

//将大端字节序转换为小端字节序
void convertToLittleEndian(unsigned int *data, int len)
{
    for (int index = 0; index < len; index ++) {
        
        *data = ((*data & 0xff000000) >> 24)
        | ((*data & 0x00ff0000) >>  8)
        | ((*data & 0x0000ff00) <<  8)
        | ((*data & 0x000000ff) << 24);
        
        data ++;
    }
}

//进行循环左移函数
void ROL(unsigned int *s, unsigned short cx)
{
    if (cx > 32)cx %= 32;
    *s = (*s << cx) | (*s >> (32 - cx));
   return;
}

//MD5加密函数
void MD5(NSString *str){
    const void * bytes[str.length];

    //字符串转字节流
    [str getBytes:bytes maxLength:str.length usedLength:nil encoding:NSUTF8StringEncoding options:NSStringEncodingConversionExternalRepresentation range:NSMakeRange(0, str.length) remainingRange:nil];
    
    //使用NSData存储
    NSMutableData * data = [[NSMutableData alloc] initWithBytes:bytes length:str.length];
    
    //进行数据填充
    BOOL first = YES;
    if (data.length<56) {   // 448 bit = 56 * 8 bit = 56 byte
        do {
            if (first) {
                int byte = 0b10000000;
                [data appendBytes:&byte length:1];
                first = NO;
            }else{
                int byte = 0b00000000;
                [data appendBytes:&byte length:1];
            }
        } while (data.length<56);
    }
    int length = (int)str.length*8%((int)pow(2, 64));
    [data appendBytes:&length length:8];
    void * newBytes[64];
    memcpy(newBytes, [data bytes], 64);

    //大小端转换
    convertToLittleEndian(newBytes, 64);
    NSData * newData = [NSData dataWithBytes:newBytes length:data.length];
    NSMutableArray * subData = [NSMutableArray array];
    
    //进行分组
    for (int i = 0; i<16; i++) {
        [subData addObject: [newData subdataWithRange:NSMakeRange(i*4, 4)]];
    }
    //初始输入
    // 四个链接变量
    unsigned int A=0x67452301,B=0xefcdab89,C=0x98badcfe,D=0x10325476;
    unsigned int a=A,b=B,c=C,d=D;
    // 每轮向左位移数
    unsigned int s[64] = { 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11,16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21 };
    // 常量ti公式:floor(abs(sin(i+1))×(2pow32)
    unsigned int  k[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
        0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
        0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
        0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391 };
    
    //64次循环处理
    for (int i = 0; i <= 64; i++) {
        
        if (i<64) {
            unsigned int f;
            unsigned int g;
            if (i < 16) {
				f = (b & c) | ((~b) & d);
				g = i;
			} else if (i < 32) {
				f = (d & b) | ((~d) & c);
				g = (5 * i + 1) % 16;
			} else if (i < 48) {
				f = b ^ c ^ d;
				g = (3 * i + 5) % 16;
			} else {
				f = c ^ (b | (~d));
				g = (7 * i) % 16;
			}
            unsigned int * temp = (unsigned int *) [subData[g] bytes];
            unsigned int  *tem = malloc(sizeof(unsigned int));
            memcpy(tem, temp, 4);
            convertToLittleEndian(tem, 4);
            unsigned int res = (a+f+*tem+k[i]);
            ROL(&res,s[i]);
            unsigned int t = res+b;
            a = d;
            d = c;
            c = b;
            b = t;
            
            // int tmp = d;
			// d = c;
			// c = b;
			// b = b + shift(a + F + K[i] + M[g], s[i]);
			// a = tmp;
        }else{
            A = a+A;
            B = b+B;
            C = c+C;
            D = d+D;
        }
    }

    //大小端转换
    unsigned int * newA = malloc(sizeof(unsigned int));
    memcpy(newA, &A, 4);

    NSLog(@"%0x",*newA);
    convertToLittleEndian(newA, 4);
    unsigned int * newB = malloc(sizeof(unsigned int));
    memcpy(newB, &B, 4);
    convertToLittleEndian(newB, 4);
    unsigned int * newC = malloc(sizeof(unsigned int));
    memcpy(newC, &C, 4);
    convertToLittleEndian(newC, 4);
    unsigned int * newD = malloc(sizeof(unsigned int));
    memcpy(newD, &D, 4);
    convertToLittleEndian(newD, 4);

    NSLog(@"AAA:%0x %0x %0x %0x ",*newA,*newB,*newC,*newD);
}
```

### 算法细节

    每一分组的算法流程如下：

    第一分组需要将四个链接变量复制到另外四个变量中：A到a，B到b，C到c，D到d。从第二分组开始的变量为上一分组的运算结果，即A = a， B = b， C = c， D = d。

    主循环有四轮，每轮循环都很相似。第一轮进行16次操作。每次操作对a、b、c和d中的其中三个作一次非线性函数运算，然后将所得结果加上第四个变量，文本的一个子分组和一个常数。再将所得结果向左环移一个不定的数，并加上a、b、c或d中之一。最后用该结果取代a、b、c或d中之一。

    以下是每次操作中用到的四个非线性函数（每轮一个）。

    F( X ,Y ,Z ) = ( X & Y ) | ( (~X) & Z )

    G( X ,Y ,Z ) = ( X & Z ) | ( Y & (~Z) )

    H( X ,Y ,Z ) =X ^ Y ^ Z

    I( X ,Y ,Z ) =Y ^ ( X | (~Z) )

    （&是与（And），|是或（Or），~是非（Not），^是异或（Xor））

    这四个函数的说明：如果X、Y和Z的对应位是独立和均匀的，那么结果的每一位也应是独立和均匀的。

    F是一个逐位运算的函数。即，如果X，那么Y，否则Z。函数H是逐位奇偶操作符。

    假设Mj表示消息的第j个子分组（从0到15），常数ti是4294967296*abs( sin(i) )的整数部分，i 取值从1到64，单位是弧度。（4294967296=2的32次方）

    现定义：

    FF(a ,b ,c ,d ,Mj ,s ,ti ) 操作为 a = b + ( (a + F(b,c,d) + Mj + ti) << s)

    GG(a ,b ,c ,d ,Mj ,s ,ti ) 操作为 a = b + ( (a + G(b,c,d) + Mj + ti) << s)

    HH(a ,b ,c ,d ,Mj ,s ,ti) 操作为 a = b + ( (a + H(b,c,d) + Mj + ti) << s)

    II(a ,b ,c ,d ,Mj ,s ,ti) 操作为 a = b + ( (a + I(b,c,d) + Mj + ti) << s)

    现定义：

    FF(a ,b ,c ,d ,Mj ,s ,ti ) 操作为 a = b + ( (a + F(b,c,d) + Mj + ti) << s)

    GG(a ,b ,c ,d ,Mj ,s ,ti ) 操作为 a = b + ( (a + G(b,c,d) + Mj + ti) << s)

    HH(a ,b ,c ,d ,Mj ,s ,ti) 操作为 a = b + ( (a + H(b,c,d) + Mj + ti) << s)

    II(a ,b ,c ,d ,Mj ,s ,ti) 操作为 a = b + ( (a + I(b,c,d) + Mj + ti) << s)

    注意：“<<”表示循环左移位，不是左移位。

    这四轮（共64步）是：

    第一轮

    FF(a ,b ,c ,d ,M0 ,7 ,0xd76aa478 )

    FF(d ,a ,b ,c ,M1 ,12 ,0xe8c7b756 )

    FF(c ,d ,a ,b ,M2 ,17 ,0x242070db )

    FF(b ,c ,d ,a ,M3 ,22 ,0xc1bdceee )

    FF(a ,b ,c ,d ,M4 ,7 ,0xf57c0faf )

    FF(d ,a ,b ,c ,M5 ,12 ,0x4787c62a )

    FF(c ,d ,a ,b ,M6 ,17 ,0xa8304613 )

    FF(b ,c ,d ,a ,M7 ,22 ,0xfd469501)

    FF(a ,b ,c ,d ,M8 ,7 ,0x698098d8 )

    FF(d ,a ,b ,c ,M9 ,12 ,0x8b44f7af )

    FF(c ,d ,a ,b ,M10 ,17 ,0xffff5bb1 )

    FF(b ,c ,d ,a ,M11 ,22 ,0x895cd7be )

    FF(a ,b ,c ,d ,M12 ,7 ,0x6b901122 )

    FF(d ,a ,b ,c ,M13 ,12 ,0xfd987193 )

    FF(c ,d ,a ,b ,M14 ,17 ,0xa679438e )

    FF(b ,c ,d ,a ,M15 ,22 ,0x49b40821 )

    第二轮

    GG(a ,b ,c ,d ,M1 ,5 ,0xf61e2562 )

    GG(d ,a ,b ,c ,M6 ,9 ,0xc040b340 )

    GG(c ,d ,a ,b ,M11 ,14 ,0x265e5a51 )

    GG(b ,c ,d ,a ,M0 ,20 ,0xe9b6c7aa )

    GG(a ,b ,c ,d ,M5 ,5 ,0xd62f105d )

    GG(d ,a ,b ,c ,M10 ,9 ,0x02441453 )

    GG(c ,d ,a ,b ,M15 ,14 ,0xd8a1e681 )

    GG(b ,c ,d ,a ,M4 ,20 ,0xe7d3fbc8 )

    GG(a ,b ,c ,d ,M9 ,5 ,0x21e1cde6 )

    GG(d ,a ,b ,c ,M14 ,9 ,0xc33707d6 )

    GG(c ,d ,a ,b ,M3 ,14 ,0xf4d50d87 )

    GG(b ,c ,d ,a ,M8 ,20 ,0x455a14ed )

    GG(a ,b ,c ,d ,M13 ,5 ,0xa9e3e905 )

    GG(d ,a ,b ,c ,M2 ,9 ,0xfcefa3f8 )

    GG(c ,d ,a ,b ,M7 ,14 ,0x676f02d9 )

    GG(b ,c ,d ,a ,M12 ,20 ,0x8d2a4c8a )

    第三轮

    HH(a ,b ,c ,d ,M5 ,4 ,0xfffa3942 )

    HH(d ,a ,b ,c ,M8 ,11 ,0x8771f681 )

    HH(c ,d ,a ,b ,M11 ,16 ,0x6d9d6122 )

    HH(b ,c ,d ,a ,M14 ,23 ,0xfde5380c )

    HH(a ,b ,c ,d ,M1 ,4 ,0xa4beea44 )

    HH(d ,a ,b ,c ,M4 ,11 ,0x4bdecfa9 )

    HH(c ,d ,a ,b ,M7 ,16 ,0xf6bb4b60 )

    HH(b ,c ,d ,a ,M10 ,23 ,0xbebfbc70 )

    HH(a ,b ,c ,d ,M13 ,4 ,0x289b7ec6 )

    HH(d ,a ,b ,c ,M0 ,11 ,0xeaa127fa )

    HH(c ,d ,a ,b ,M3 ,16 ,0xd4ef3085 )

    HH(b ,c ,d ,a ,M6 ,23 ,0x04881d05 )

    HH(a ,b ,c ,d ,M9 ,4 ,0xd9d4d039 )

    HH(d ,a ,b ,c ,M12 ,11 ,0xe6db99e5 )

    HH(c ,d ,a ,b ,M15 ,16 ,0x1fa27cf8 )

    HH(b ,c ,d ,a ,M2 ,23 ,0xc4ac5665 )

    第四轮

    II(a ,b ,c ,d ,M0 ,6 ,0xf4292244 )

    II(d ,a ,b ,c ,M7 ,10 ,0x432aff97 )

    II(c ,d ,a ,b ,M14 ,15 ,0xab9423a7 )

    II(b ,c ,d ,a ,M5 ,21 ,0xfc93a039 )

    II(a ,b ,c ,d ,M12 ,6 ,0x655b59c3 )

    II(d ,a ,b ,c ,M3 ,10 ,0x8f0ccc92 )

    II(c ,d ,a ,b ,M10 ,15 ,0xffeff47d )

    II(b ,c ,d ,a ,M1 ,21 ,0x85845dd1 )

    II(a ,b ,c ,d ,M8 ,6 ,0x6fa87e4f )

    II(d ,a ,b ,c ,M15 ,10 ,0xfe2ce6e0 )

    II(c ,d ,a ,b ,M6 ,15 ,0xa3014314 )

    II(b ,c ,d ,a ,M13 ,21 ,0x4e0811a1 )

    II(a ,b ,c ,d ,M4 ,6 ,0xf7537e82 )

    II(d ,a ,b ,c ,M11 ,10 ,0xbd3af235 )

    II(c ,d ,a ,b ,M2 ,15 ,0x2ad7d2bb )

    II(b ,c ,d ,a ,M9 ,21 ,0xeb86d391 )

    所有这些完成之后，将a、b、c、d分别在原来基础上再加上A、B、C、D。

    即a = a + A，b = b + B，c = c + C，d = d + D

    然后用下一分组数据继续运行以上算法。