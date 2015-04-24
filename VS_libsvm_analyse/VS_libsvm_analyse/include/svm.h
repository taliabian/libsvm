#ifndef _LIBSVM_H
#define _LIBSVM_H

#define LIBSVM_VERSION 320

#ifdef __cplusplus
extern "C" { /// C和C++混合编程
#endif

extern int libsvm_version;

/// 存储单一向量中的单个特征
struct svm_node
{
	int index;
	double value;
};
/// 存储参加本次运算的所有样本(数据集),及其所属类别
struct svm_problem
{
	int l; /*样本总数*/
	double *y; /*指向样本所属类别的数组*/
	struct svm_node **x;/*指向存储内容为指针的数组*/
};

//  0 -- C-SVC		(multi-class classification) 
//	1 -- nu-SVC		(multi-class classification)
//	2 -- one-class SVM	
//	3 -- epsilon-SVR	(regression)
//	4 -- nu-SVR		(regression)
enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type (default 0)*/

//	LINEAR线性核 K( x(i),x(j) ) = x(i)^T*x(j)
//	POLY多项式核 K( x(i),x(j) ) = ( gamma*x(i)^T * x(j) + coef0 )^degree, gamma>0
//	RBF核 K( x(i),x(j) ) = exp( -gamma * || x(i)-x(j) ||^2 ), gamma>0 
//	SIGMOID核 K( x(i),x(j) ) = tanh( gamma * x(i)^T*x(j) + coef0 )
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type (default 2) */

struct svm_parameter
{
	int svm_type;
	int kernel_type;
	int degree;	/* for poly 多项式核中的degree (default 3)*/
	double gamma;	/* for poly/rbf/sigmoid 多项式核/rbf/sigmod核中的gamma (default 1/num_features)*/
	double coef0;	/* for poly/sigmoid 多项式核/sigmod核中的coef0 (default 0)*/

	/* these are for training only */
	double cache_size; /* in MB 训练所需内存 (default 100)*/
	double eps;	/* stopping criteria 设置终止准则中的可容忍偏差(default 0.001)  */
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR  ( 惩罚系数C default 1, C越大越训练越复杂, 耗时越久)*/
	int nr_weight;		/* for C_SVC 权重数目*/
	int *weight_label;	/* for C_SVC 权重 */
	double* weight;		/* for C_SVC (default 1)*/
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR ( default 0.5)*/
	double p;	/* for EPSILON_SVR ( default 0.1 )*/
	int shrinking;	/* use the shrinking heuristics 训练过程是否使用压缩( default 1 )*/
	int probability; /* do probability estimates  是否做概率估计(default 0)*/
};

//
// svm_model 保存训练后的训练模型
// 
struct svm_model
{
	struct svm_parameter param;	/* parameter 训练参数 */
	int nr_class;		/* number of classes, = 2 in regression/one class svm 类别数 */
	int l;			/* total #SV 支持向量数 */
	struct svm_node **SV;/* SVs (SV[l]) 保存支持向量的指针，至于支持向量的内容，
						如果是从文件中读取，内容会额外保留；如果是直接训练得来，
						则保留在原来的训练集中。如果训练完成后需要预报，原来的
						训练集内存不可以释放*/
	double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l])
						相当于判别函数中的alpha */
	double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) 
						相当于判别函数中的b*/
	double *probA;		/* pariwise probability information */
	double *probB;
	int *sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

	/* for classification only */

	int *label;		/* label of each class (label[k]) */
	int *nSV;		/* number of SVs for each class (nSV[k]) */
				/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
	/* XXX */
	int free_sv;		/* 1 if svm_model is created by svm_load_model*/
				/* 0 if svm_model is created by svm_train */
};

/// 最主要的驱动函数， 训练数据
struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);

/// 用SVM做交叉验证
void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

/// 保存训练好的模型文件
int svm_save_model(const char *model_file_name, const struct svm_model *model);

/// 从文件中把训练好的模型读到内存中
struct svm_model *svm_load_model(const char *model_file_name);

/// 获得数据集的svm_type
int svm_get_svm_type(const struct svm_model *model);

/// 获得数据集的类别数(必须经过训练得到模型后才可以用)
int svm_get_nr_class(const struct svm_model *model);

/// 获得数据集的类别号(必须经过训练得到模型后才可以用)
void svm_get_labels(const struct svm_model *model, int *label);

void svm_get_sv_indices(const struct svm_model *model, int *sv_indices);

int svm_get_nr_sv(const struct svm_model *model);

double svm_get_svr_probability(const struct svm_model *model);

/// 用训练好的模型预报样本的值，输出结果保留到数组中(非接口函数 )
double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);

/// 预测某一样本的值
double svm_predict(const struct svm_model *model, const struct svm_node *x);

double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

void svm_free_model_content(struct svm_model *model_ptr);

/// 消除训练模型,释放资源
void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);

void svm_destroy_param(struct svm_parameter *param);

/// 检查输入参数，保证后面的训练能正常进行
const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);

int svm_check_probability_model(const struct svm_model *model);

void svm_set_print_string_function(void (*print_func)(const char *));

#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */
