package ML::CNN;
use Modern::Perl;
# based on: https://zzutk.github.io/docs/reports/2016.10%20-%20Derivation%20of%20Backpropagation%20in%20Convolutional%20Neural%20Network%20(CNN).pdf
use Math::Random qw(random_uniform);
use Data::Dumper;
use ML::Util qw(print_2d_array matmul transpose print_1d_array rotate_matrix_180 conv2d add_2_arrays);
use Cwd qw(abs_path);
use JSON;


my $kernel_limit = 15;
my $c1_kernels_limit = 15;

my $code;
BEGIN {
   $code = <<'EOCODE';
// This section is boilerplace code to move data from Perl -> C and back again

#define HAVE_PERL_VERSION(R, V, S) \
    (PERL_REVISION > (R) || (PERL_REVISION == (R) && (PERL_VERSION > (V) || (PERL_VERSION == (V) && (PERL_SUBVERSION >= (S))))))

#define sv_setrv(s, r)  S_sv_setrv(aTHX_ s, r)

static void S_sv_setrv(pTHX_ SV *sv, SV *rv)
{
  sv_setiv(sv, (IV)rv);
#if !HAVE_PERL_VERSION(5, 24, 0)
  SvIOK_off(sv);
#endif
  SvROK_on(sv);
}

int is_array_ref(
        SV *array,
        size_t *array_sz
);
int array_numelts_2D(
        SV *array,
        size_t *_Nd1,
        size_t **_Nd2
);
int array_of_unsigned_int_into_AV(
        size_t *src,
        size_t src_sz,
        SV *dst
);
int array_of_int_into_AV(
        int *src,
        size_t src_sz,
        SV *dst
);

int is_array_ref(
        SV *array,
        size_t *array_sz
){
        if( ! SvROK(array) ){ fprintf(stderr, "is_array_ref() : warning, input '%p' is not a reference.\n", array); return 0; }
        if( SvTYPE(SvRV(array)) != SVt_PVAV ){ fprintf(stderr, "is_array_ref() : warning, input ref '%p' is not an ARRAY reference.\n", array); return 0; }
        // it's an array, cast it to AV to get its len via av_len();
        // yes, av_len needs to be bumped up
        int asz = 1+av_len((AV *)SvRV(array));
        if( asz < 0 ){ fprintf(stderr, "is_array_ref() : error, input array ref '%p' has negative size!\n", array); return 0; }
        *array_sz = (size_t )asz;
        return 1; // success, it is an array and size returned by ref, above
}

#define array_numelts_1D(A,B) (!is_array_ref(A,B))

int array_numelts_2D(
        SV *array,
        size_t *_Nd1,
        size_t **_Nd2
){
        size_t anN, anN2, *Nd2 = NULL;

        if( ! is_array_ref(array, &anN) ){
           fprintf(stderr, "is_array_ref_2D() : error, call to is_array_ref() has failed for array '%p'.\n", array);
           return 1;
        }

        if( *_Nd2 == NULL ){
           if( (Nd2=(size_t *)malloc(anN*sizeof(size_t))) == NULL ){
               fprintf(stderr, "array_numelts_2D() : error, failed to allocate %zu bytes for %zu items for Nd2.\n", anN*sizeof(size_t), anN);
               return 1;
           }
        } else Nd2 = *_Nd2;
        AV *anAV = (AV *)SvRV(array);
        size_t *pNd2 = &(Nd2[0]);
        for(size_t i=0;i<anN;i++,pNd2++){
           SV *subarray = *av_fetch(anAV, i, FALSE);
           if( ! is_array_ref(subarray, &anN2) ){
              fprintf(stderr, "is_array_ref_2D() : error, call to is_array_ref() has failed for [%p][%p], item %zu.\n", array, subarray, i);
              if(*_Nd2==NULL) free(Nd2);
              return 1;
           }
           *pNd2 = anN2;
        }
        if( *_Nd2 == NULL ) *_Nd2 = Nd2;
        *_Nd1 = anN;
        return 0; // success
}

int array_of_int_into_AV(
        int *src,
        size_t src_sz,
        SV *dst
){
        size_t dst_sz;
        if( ! is_array_ref(dst, &dst_sz) ){ fprintf(stderr, "array_of_int_into_AV() : error, call to is_array_ref() has failed.\n"); return 1; }
        AV *dstAV = (AV *)SvRV(dst);
        for(size_t i=0;i<src_sz;i++){
                av_push(dstAV, newSViv(src[i]));
        }
        return 0; // success
}

// end of Perl -> C -> Perl section

float ** c1_kernels;
float ** delta_k1;
float ** rot_c1_kernels;
float ** c2_kernels;
float ** delta_c2_kernels;
float ** rot_c2_kernels;
float ** s1_pooled;
float ** delta_p1;
float ** rot_s1_pooled;
float ** s2_pooled;
float ** delta_p2;
float ** c1;
float ** delta_c1;
float ** rot_delta_c1;
float ** c2;
float ** delta_c2;
float ** delta_c2_sigmoid;
float ** rot_delta_c2_sigmoid;
float * f;
float * f_T;
float * delta_f;
float * y;
float loss;
float * label;
float * delta_y;
float * delta_fc_bias;
float * delta_W;

float * input;
float * rot_input;


float * c1_biases;
float * delta_b1;
float * c2_biases;
float * delta_b2;
float * fc_biases;
float * fc_weights;
float * W_T;

int debug = 0;

void set_debug() {
  debug = 1;
}

void unset_debug() {
  debug = 0;
}

int kernel_size, c1_kernel_count, c2_kernel_count, input_size, c1_size, s1_size, s2_size, c2_size, fc_input_size, fc_output_size, batch_size;

void init_params(int ks, int p, int q, int insize, int c1size, int s1size, int s2size, int c2size, int fcis, int fcos ) {
   kernel_size = ks;
   c1_kernel_count = p;
   c2_kernel_count = q;
   input_size = insize;
   c1_size = c1size;
   s1_size = s1size;
   s2_size = s2size;
   for (int i=0;i<c1_kernel_count;i++) {
       c1[i] = (float *)malloc(sizeof(float)*c1_size*c1_size);
       delta_c1[i] = (float *)malloc(sizeof(float)*c1_size*c1_size);
       rot_delta_c1[i] = (float *)malloc(sizeof(float)*c1_size*c1_size);
       s1_pooled[i] = (float *)malloc(sizeof(float)*s1_size*s1_size);
       rot_s1_pooled[i] = (float *)malloc(sizeof(float)*s1_size*s1_size);
       delta_p1[i] = (float *)malloc(sizeof(float)*s1_size*s1_size);
   }
   c2_size = c2size;
   for (int i=0;i<c2_kernel_count;i++) {
       c2[i] = (float *)malloc(sizeof(float)*c2_size*c2_size);
       s2_pooled[i] = (float *)malloc(sizeof(float)*s2_size*s2_size);
       delta_p2[i] = (float *)malloc(sizeof(float)*s2_size*s2_size);
       delta_c2[i] = (float *)malloc(sizeof(float)*c2_size*c2_size);
       delta_c2_sigmoid[i] = (float *)malloc(sizeof(float)*c2_size*c2_size);
       rot_delta_c2_sigmoid[i] = (float *)malloc(sizeof(float)*c2_size*c2_size);
   }
   fc_input_size = fcis;
   fc_output_size = fcos;
   batch_size = 1;
   f = (float *)malloc(sizeof(float)*fc_input_size*batch_size);
   delta_f = (float *)malloc(sizeof(float)*fc_input_size*batch_size);
   f_T = (float *)malloc(sizeof(float)*fc_input_size*batch_size);
   input  = (float *)malloc(sizeof(float)*input_size*input_size);
   rot_input  = (float *)malloc(sizeof(float)*input_size*input_size);
   y  = (float *)malloc(sizeof(float)*fc_output_size);
   label  = (float *)malloc(sizeof(float)*fc_output_size);
   delta_y  = (float *)malloc(sizeof(float)*fc_output_size*batch_size);
   delta_fc_bias = (float *)malloc(sizeof(float)*fc_output_size);
   fc_weights = (float *)malloc(sizeof(float)*fc_input_size*fc_output_size);
   delta_W = (float *)malloc(sizeof(float)*fc_input_size*fc_output_size);
   W_T = (float *)malloc(sizeof(float)*fc_input_size*fc_output_size);
}

void rotate_180(int arr_size, float * inarr, float * outarr) {
   int i,j;
   for (i=0;i<arr_size;i++) {
      for (j=0;j<arr_size;j++) {
          outarr[ ((arr_size - i - 1) * arr_size) + (arr_size - j - 1)] = inarr[i * arr_size + j];
      }
   }
}

void init_c1_kernels(int p) {
   c1_kernels = (float **)malloc(p * sizeof(float*));
   delta_k1 = (float **)malloc(p * sizeof(float*));
   rot_c1_kernels = (float **)malloc(p * sizeof(float*));
   c1 = (float **)malloc(p * sizeof(float*));
   delta_c1 = (float **)malloc(p * sizeof(float*));
   rot_delta_c1 = (float **)malloc(p * sizeof(float*));
   s1_pooled = (float **)malloc(p * sizeof(float*));
   delta_p1 = (float **)malloc(p * sizeof(float*));
   rot_s1_pooled = (float **)malloc(p * sizeof(float*));
}


void init_c2_kernels(int q, int p) {
   c2_kernels = (float **)malloc(p*q*sizeof(float*));
   rot_c2_kernels = (float **)malloc(p*q*sizeof(float*));
   delta_c2_kernels = (float **)malloc(p*q*sizeof(float*));
   if (debug == 1) {
      printf("init c2 with %i entries\n", q);
   }
   c2 = (float **)malloc(q * sizeof(float*));
   delta_c2 = (float **)malloc(q * sizeof(float*));
   delta_c2_sigmoid = (float **)malloc(q * sizeof(float*));
   rot_delta_c2_sigmoid = (float **)malloc(q * sizeof(float*));
   s2_pooled = (float **)malloc(q * sizeof(float*));
   delta_p2 = (float **)malloc(q * sizeof(float*));
}

void set_c1_biases(SV *b) {
   size_t AH,i;
   AV *av;
   SV *subav;
   float *pd;
   array_numelts_1D(b,&AH);
   c1_biases = (float *)malloc(sizeof(float)*AH);
   delta_b1 = (float *)malloc(sizeof(float)*AH);
   pd = c1_biases;
   av = (AV *)SvRV(b);
   for(i=0;i<AH;i++) {
      subav = *av_fetch(av, i, FALSE);
      *pd = SvNV(subav);
      pd++;
   }
}

void set_c2_biases(SV *b) {
   size_t AH,i;
   AV *av;
   SV *subav;
   float *pd;
   array_numelts_1D(b,&AH);
   c2_biases = (float *)malloc(sizeof(float)*AH);
   delta_b2 = (float *)malloc(sizeof(float)*AH);
   pd = c2_biases;
   av = (AV *)SvRV(b);
   for(i=0;i<AH;i++) {
      subav = *av_fetch(av, i, FALSE);
      *pd = SvNV(subav);
      pd++;
   }
}

void set_fc_biases(SV *b) {
   size_t AH,i;
   AV *av;
   SV *subav;
   float *pd;
   array_numelts_1D(b,&AH);
   fc_biases = (float *)malloc(sizeof(float)*AH);
   pd = fc_biases;
   av = (AV *)SvRV(b);
   for(i=0;i<AH;i++) {
      subav = *av_fetch(av, i, FALSE);
      *pd = SvNV(subav);
      pd++;
   }
}

void add_c1_kernel(int p, SV *k) {
// convert Perl weight array to C array of floats 
   AV *av;
   float *pd;
   size_t i,j;
   size_t AH, AW, *AWs = NULL;
   SV *subav, *subsubav;

   array_numelts_2D(k, &AH, &AWs);
   AW = AWs[0];

   c1_kernels[p] = (float *)malloc(sizeof(float)*AW*AH);
   delta_k1[p] = (float *)malloc(sizeof(float)*AW*AH);
   rot_c1_kernels[p] = (float *)malloc(sizeof(float)*AW*AH);
   pd = c1_kernels[p];
   av = (AV *)SvRV(k);
   for(i=0;i<AH;i++){ // for each row
      subav = *av_fetch(av, i, FALSE);
      for(j=0;j<AW;j++){ // for the cols of that row
         subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
         *pd = SvNV(subsubav);
         pd++;
      }
   }
   rotate_180(AH, c1_kernels[p], rot_c1_kernels[p]);
}

void add_c2_kernel(int q, int p, int colsize, SV *k) {
// convert Perl weight array to C array of floats
    AV *av;
    float *pd;
    size_t i,j;
    size_t AH, AW, *AWs = NULL;
    SV *subav, *subsubav;

    array_numelts_2D(k, &AH, &AWs);
    AW = AWs[0];

    c2_kernels[q * colsize + p] = (float *)malloc(sizeof(float)*AW*AH);
    rot_c2_kernels[q * colsize + p] = (float *)malloc(sizeof(float)*AW*AH);
    delta_c2_kernels[q * colsize + p] = (float *)malloc(sizeof(float)*AW*AH);
    pd = c2_kernels[q * colsize + p];
    av = (AV *)SvRV(k);
    for(i=0;i<AH;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<AW;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }
   rotate_180(AH, c2_kernels[q * colsize + p], rot_c2_kernels[q * colsize + p]);
}

void set_fc_weights(SV *w) {
// convert Perl weight array to C array of floats
    AV *av;
    float *pd;
    size_t i,j;
    size_t AH, AW, *AWs = NULL;
    SV *subav, *subsubav;

    array_numelts_2D(w, &AH, &AWs);
    AW = AWs[0];

    pd = fc_weights;
    av = (AV *)SvRV(w);
    for(i=0;i<AH;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<AW;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }
}

void load_input (SV *w) {
    AV *av;
    float *pd;
    size_t i,j;
    size_t AH, AW, *AWs = NULL;
    SV *subav, *subsubav;

    array_numelts_2D(w, &AH, &AWs);
    AW = AWs[0];

    pd = input;
    av = (AV *)SvRV(w);
    for(i=0;i<AH;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<AW;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }
}

void set_label (SV *l) {
    AV *av;
    float *pd;
    size_t i,j;
    size_t AH, AW, *AWs = NULL;
    SV *subav, *subsubav;

    array_numelts_2D(l, &AH, &AWs);
    AW = AWs[0];

    pd = label;
    av = (AV *)SvRV(l);
    for(i=0;i<AH;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<AW;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }
}

void print_2D_array(float *foo, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%+.5f\t", foo[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}


void conv2d(float * outarr, float * inarr, int inarr_size, float * kernel, int k_size, char * action) {
// does not rotate kernel, i.e. assumes a rotated kernel is passed in
   int output_size = inarr_size;
   int kernel_offset = k_size  / 2;
   int in_offset = 0;

   if (!strcmp(action, "reduce")) {
     // std::cout << "reduce" << std::endl;
      output_size -= kernel_offset * 2 ;
      if (k_size % 2 == 0) {
         output_size++;
      }
   } else if (!strcmp(action, "expand")) {
      output_size += kernel_offset * 2;
      in_offset = kernel_offset * 2;
   } else {
      std::cout << "not a valid action" << std::endl;
   }

   for (int i = 0; i < output_size;i++) {
      for (int j = 0; j < output_size;j++) {
         outarr[i*output_size + j] = 0;
         for (int u = 0; u < k_size; u++) {
            for (int v = 0; v < k_size; v++) {
               int in_i = i + u - in_offset;
               int in_j = j + v - in_offset;
               if (in_j >= 0 && in_i >= 0
                   && in_i < inarr_size && in_j < inarr_size) {
                  outarr[i * output_size + j] += kernel[u * k_size + v] * inarr[in_i * inarr_size + in_j];
               }
            }
         }
      }
   }
}

void conv2d_activation(float * inarr, int inarr_size, float bias) {
   for (int i = 0; i < inarr_size;i++) {
      for (int j = 0; j < inarr_size;j++) {
         inarr[i*inarr_size + j] = 1 / (1 + exp(-1 * (inarr[i*inarr_size + j] + bias)));
      }
   }
}


void c1_conv () {
   for (int i=0;i<c1_kernel_count;i++) {
      if (debug == 1) {
         std::cout << " c1 before convolution " << i << std::endl;
         print_2D_array(c1[i], c1_size, c1_size);
         std::cout << "c1 kernel " << i << std::endl;
         print_2D_array(c1_kernels[i], kernel_size, kernel_size);
         std::cout << "rotated c1 kernel " << i << std::endl;
         print_2D_array(rot_c1_kernels[i], kernel_size, kernel_size);
      }
      conv2d(c1[i], input, input_size, rot_c1_kernels[i], kernel_size, "reduce");
      if (debug == 1) {
         std::cout << " c1 before activation " << i << std::endl;
         print_2D_array(c1[i], c1_size, c1_size);
      }
      conv2d_activation(c1[i], c1_size, c1_biases[i]);
      if (debug == 1) {
         std::cout << " c1 after activation " << i << std::endl;
         print_2D_array(c1[i], c1_size, c1_size);
      }
  } 
}

void add_2_arrays(float * basearr, float * addarr, int rows, int cols) {
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         basearr[i*rows + j] += addarr[i*rows + j];
      }
   }
}

void c2_conv () {

   float * c2_work_area = (float *)malloc(sizeof(float) * c2_size * c2_size);
   for (int q=0;q<c2_kernel_count;q++) {
      for (int i = 0; i < c2_size; i++) {
         for (int j = 0; j < c2_size; j++) {
            c2[q][i * c2_size + j] = 0;
         }
      }
      for (int p=0;p<c1_kernel_count;p++) {
         conv2d(c2_work_area, s1_pooled[p], s1_size, rot_c2_kernels[q*c1_kernel_count + p], kernel_size, "reduce");
         add_2_arrays(c2[q], c2_work_area, c2_size, c2_size);
      }
      if (debug == 1) {
         std::cout << " c2 before activation " << q << std::endl;
         print_2D_array(c2[q], c2_size, c2_size);
      }
      conv2d_activation(c2[q], c2_size, c2_biases[q]);
      if (debug == 1) {
         std::cout << " c2 after activation " << q << std::endl;
         print_2D_array(c2[q], c2_size, c2_size);
      }
  } 
  free(c2_work_area);
}

void c1_pooling() {
   for (int p=0;p<c1_kernel_count;p++) {
      for (int i=0;i<s1_size;i++) {
         for (int j=0;j<s1_size;j++) {
            s1_pooled[p][i * s1_size + j] = 0;
            for (int v=0;v<2;v++) { 
               for (int u=0;u<2;u++) { 
// algorithm defined as 1 relative, but perl arrays are 0 relative, so it is 2i + u, rather than 2i - u etc
                  s1_pooled[p][i * s1_size + j] += c1[p][ (2 * i + u) * c1_size + (2 * j + v)]/4;
               }
            }
         }
      }
      if (debug == 1) {
         std::cout << " p1 pooled " << p << std::endl;
         print_2D_array(s1_pooled[p], s1_size, s1_size);
      }
   }
}

void c2_pooling() {
   for (int q=0;q<c2_kernel_count;q++) {
      for (int i=0;i<s2_size;i++) {
         for (int j=0;j<s2_size;j++) {
            s2_pooled[q][i * s2_size + j] = 0;
            for (int v=0;v<2;v++) {
               for (int u=0;u<2;u++) {
// algorithm defined as 1 relative, but perl arrays are 0 relative, so it is 2i + u, rather than 2i - u etc
                  s2_pooled[q][i * s2_size + j] += c2[q][ (2 * i + u) * c2_size + (2 * j + v)]/4;
               }
            }
         }
      }
      if (debug == 1) {
         std::cout << " p2 pooled " << q << std::endl;
         print_2D_array(s2_pooled[q], s2_size, s2_size);
      }
   }
}

void vectorise() {
   for (int q = 0;q<c2_kernel_count;q++) {
      for (int j = 0;j<s2_size;j++) {
         for (int i = 0;i<s2_size;i++) {
// each output must be 1 item per column, not 1 item per row.  So when we have a batch the matrix will be "batch entries" wide
            f[ (q * s2_size * s2_size) + (j * s2_size) + i] = s2_pooled[q][i * s2_size + j];
            //push @{$self->{f}[$row++]}, $self->{p2}[$q][$i][$j];
         }
      }
   }
   if (debug == 1) {
      std::cout << " f " << std::endl;
      print_2D_array(f, fc_input_size, 1);
   }
}

void final_connected_layer() {
   for (int i = 0;i < fc_output_size; i++) {
      y[i] = 0;
   }
   //fc_input_cols = 1, fc_input rows = 192, w_cols = 192, w rows = 10
   for (int i = 0; i < fc_output_size; i++ ) {
      for (int j = 0; j < batch_size; j++ ) {
         for (int k = 0; k < fc_input_size; k++ ) {
            y[i] += fc_weights[i * fc_input_size + k] * f[j * fc_input_size + k];
         }
      }
   }
   if (debug == 1) {
      std::cout << " y before sigmoid " << std::endl;
      print_2D_array(y, fc_output_size, 1);
   }
   for (int i = 0;i < fc_output_size; i++) {
       y[i] = 1 / (1 + exp(-1 * (y[i] + fc_biases[i])));
   }
   if (debug == 1) {
      std::cout << " fc_biases " << std::endl;
      print_2D_array(fc_biases, fc_output_size, 1);
      std::cout << " y " << std::endl;
      print_2D_array(y, fc_output_size, 1);
   }
   
}

float calculate_loss() {
  loss = 0;
  for (int i=0;i<fc_output_size;i++) {
     for (int j=0;j<batch_size;j++) {
        loss += 0.5 * pow(y[i * batch_size + j] - label[i * batch_size + j],2);
     }
   }
   return loss;
}

void fc_delta() {
   for (int i=0;i<fc_output_size;i++) {
      for (int j=0;j<batch_size;j++) {
         delta_y[i * batch_size + j] = (y[i * batch_size + j] - label[i * batch_size + j]) * y[i * batch_size + j] * ( 1 - y[i * batch_size + j]);
      }
   } 
   if (debug == 1) {
      printf("y\n");
      print_2D_array(y, fc_output_size, batch_size);
      printf("label\n");
      print_2D_array(label, fc_output_size, batch_size);
      printf("delta_y\n");
      print_2D_array(delta_y, fc_output_size, batch_size);
   }
}

void fc_bias_delta() {
   for (int i=0;i<fc_output_size;i++) {
      for (int j=0;j<batch_size;j++) {
         delta_fc_bias[i] = delta_y[i * batch_size + j] / batch_size;
      }
   } 
}   

void transpose_matrix(float * outarr, float * inarr, int inrows, int incols) {
   for (int i=0;i<inrows;i++) {
      for (int j=0;j<incols;j++) {
         outarr[j * inrows + i] = inarr[i * incols + j];
      }
   }
}

void w_delta() {
// delta_y x f_T = (fc_output_size, batch_size) x (batch_size, fc_input_size) = (fc_output_size, fc_input_size)
   transpose_matrix(f_T, f, batch_size, fc_input_size );

   for (int i = 0;i < fc_output_size; i++) {
      for (int j = 0;j < fc_input_size; j++) {
         delta_W[i * fc_input_size + j] = 0;
      }
   }
   for (int i = 0; i < fc_output_size; i++ ) {
      for (int j = 0; j < fc_input_size; j++ ) {
         for (int k = 0; k < batch_size; k++ ) {
            delta_W[i * fc_input_size + j] += delta_y[i * batch_size + k] * f_T[k * fc_input_size + j];
         }
      }
   }
   if (debug == 1) {
      std::cout << " delta_W " << std::endl;
      print_2D_array(delta_W, fc_output_size, fc_input_size);
   }
}

void f_delta() {
   transpose_matrix(W_T, fc_weights, fc_output_size, fc_input_size);
   if (debug == 1) {
      std::cout << " W_T " << std::endl;
      print_2D_array(W_T, fc_input_size, fc_output_size);
   }

   for (int i = 0;i < fc_input_size; i++) {
      for (int j = 0;j < batch_size; j++) {
         delta_f[i * batch_size + j] = 0;
      }
   }
   for (int i = 0; i < fc_input_size; i++ ) {
      for (int j = 0; j < batch_size; j++ ) {
         for (int k = 0; k < fc_output_size; k++ ) {
            delta_f[i * batch_size + j] += W_T[i * fc_output_size + k] * delta_fc_bias[k];
         }
      }
   }
   if (debug == 1) {
      std::cout << " delta_f " << std::endl;
      print_2D_array(delta_f, fc_input_size, batch_size);
   }
}

void p2_delta() {

   for (int q = 0; q < c2_kernel_count;q++) {
      for (int j = 0; j < s2_size; j++) {
         for (int i = 0; i < s2_size; i++) {
            delta_p2[q][i * s2_size + j] = delta_f[(q * s2_size * s2_size) + j * s2_size + i];
         }
      }
      if (debug == 1) {
         std::cout << " delta_p2 " << q << std::endl;
         print_2D_array(delta_p2[q], s2_size, s2_size);
      }
   }
}

void c2_delta() {
   for (int q = 0;q<c2_kernel_count;q++) {
      for (int i=0;i<c2_size;i++) {
         for (int j=0;j<c2_size;j++) {
            delta_c2[q][i*c2_size+j] = 0;
         }
      }
   }
   for (int q = 0;q<c2_kernel_count;q++) {
      for (int i=0;i<c2_size;i++) {
         for (int j=0;j<c2_size;j++) {
             delta_c2[q][i*c2_size+j] += 0.25 * delta_p2[q][(i/2)*s2_size + j/2];
         }
      }
      for (int i=0;i<c2_size;i++) {
         for (int j=0;j<c2_size;j++) {
             delta_c2_sigmoid[q][i*c2_size+j] = delta_c2[q][i*c2_size+j] * c2[q][i*c2_size+j] * ( 1 - c2[q][i*c2_size+j] );
         }
      }
      if (debug == 1) {
         std::cout << " delta_c2 " << q << std::endl;
         print_2D_array(delta_c2[q], c2_size, c2_size);
         std::cout << " delta_c2_sigmoid " << q << std::endl;
         print_2D_array(delta_c2_sigmoid[q], c2_size, c2_size);
      }
   }
}

void k2_delta() {
   for (int p=0;p<c1_kernel_count;p++) {
      rotate_180(s1_size, s1_pooled[p], rot_s1_pooled[p]); 
   }
   for (int q=0;q<c2_kernel_count;q++) {
      rotate_180(c2_size, delta_c2_sigmoid[q], rot_delta_c2_sigmoid[q]); 
   }
   for (int q=0;q<c2_kernel_count;q++) {
      for (int p=0;p<c1_kernel_count;p++) {
         conv2d(delta_c2_kernels[q * c1_kernel_count + p], rot_s1_pooled[p], s1_size, rot_delta_c2_sigmoid[q],c2_size,"reduce");
         if (debug == 1) {
            std::cout << " rot_s1_pooled " << p << std::endl << std::endl ;
            print_2D_array(rot_s1_pooled[p], s1_size, s1_size);
            std::cout << " delta_c2_sigmoid " << q << std::endl << std::endl ;
            print_2D_array(delta_c2_sigmoid[q], c2_size, c2_size);
            std::cout << " delta_c2_kernel " << q << "/" << p << std::endl << std::endl ;
            print_2D_array(delta_c2_kernels[q * c1_kernel_count + p], kernel_size, kernel_size);
         }
      }
   }
}

void b2_delta() {
   for (int q = 0; q < c2_kernel_count; q++) {
      delta_b2[q] = 0;
      for (int i = 0; i < c2_size; i++) {
         for (int j = 0; j < c2_size; j++) {
            delta_b2[q] += delta_c2_sigmoid[q][i * c2_size + j];
if (debug == 1) {
std::cout << "delta_b2 " << q << " adding delta_c2_sigmoid " << q << " " << i  << " " << j << " " << delta_c2_sigmoid[q][i * c2_size + j] << " = " << delta_b2[q] << std::endl;
} 
         }
      }
   }
   if (debug == 1) {
      std::cout << " delta_b2 " << std::endl ;
      print_2D_array(delta_b2, c2_kernel_count, 1);
   }

}

void p1_delta() {
   float * p1_work_area = (float *)malloc(sizeof(float)*s1_size*s1_size);
   for (int p = 0;p<c1_kernel_count;p++) {
      for (int i = 0;i < s1_size; i++) {
         for (int j = 0;j < s1_size; j++) {
            delta_p1[p][i * s1_size + j] = 0;
         }
      }
      for (int q = 0;q < c2_kernel_count;q++) {
         conv2d(p1_work_area, delta_c2_sigmoid[q], c2_size, c2_kernels[ q*c1_kernel_count + p], kernel_size, "expand");
         add_2_arrays(delta_p1[p], p1_work_area, s1_size, s1_size);
      }
      if (debug == 1) {
         std::cout << " delta_p1 " << p << std::endl ;
         print_2D_array(delta_p1[p], s1_size, s1_size);
      }
   }
}

void c1_delta() {
   for (int p = 0;p<c1_kernel_count;p++) {
      for (int i = 0;i < c1_size; i++) {
         for (int j = 0;j < c1_size; j++) {
            delta_c1[p][i * c1_size + j] = 0.25 * delta_p1[p][(i/2 * s1_size) + j/2];
         }
      }
      if (debug == 1) {
         std::cout << " delta_c1 " << p << std::endl ;
         print_2D_array(delta_c1[p], c1_size, c1_size);
      }
   }    
}

void c1_delta_sigmoid() {
   for (int p = 0;p<c1_kernel_count;p++) {
      for (int i = 0;i < c1_size; i++) {
         for (int j = 0;j < c1_size; j++) {
            delta_c1[p][i * c1_size + j] = delta_c1[p][i * c1_size + j] * c1[p][i * c1_size + j] * ( 1 - c1[p][i * c1_size + j] );
         }
      }
      rotate_180(c1_size, delta_c1[p], rot_delta_c1[p]);
      if (debug == 1) {
         std::cout << " delta_c1 (after sigmoid)" << p << std::endl << std::endl;
         print_2D_array(delta_c1[p], c1_size, c1_size);
      }
   }    
}

void k1_delta() {
   rotate_180(input_size, input, rot_input);
   for (int p=0;p<c1_kernel_count;p++) {
      conv2d(delta_k1[p], rot_input, input_size, rot_delta_c1[p], c1_size, "reduce");
      if (debug == 1) {
         std::cout << " delta_k1 " << p << std::endl << std::endl;
         print_2D_array(delta_k1[p], kernel_size, kernel_size);
      }
   }
}

void b1_delta() {
   for (int p=0;p<c1_kernel_count;p++) {
      delta_b1[p] = 0;
      for (int i=0;i<c1_size;i++) {
         for (int j=0;j<c1_size;j++) {
            delta_b1[p] += delta_c1[p][i * c1_size + j];
         }
      }
   }
   if (debug == 1) {
      std::cout << " delta_b1 " << std::endl;
      print_2D_array(delta_b1, c1_kernel_count, 1);
   }
}

void update_params( float learning_rate, float decay) {
   for (int p=0;p<c1_kernel_count;p++) {
      for (int x=0;x<kernel_size;x++) {
         for (int y=0;y<kernel_size;y++) {
            c1_kernels[p][x * kernel_size + y] -= learning_rate * delta_k1[p][x * kernel_size + y];
         }
      }
      if (debug == 1) {
         printf("c1 kernels %i\n", p);
         print_2D_array(c1_kernels[p], kernel_size, kernel_size);
      }
      rotate_180(kernel_size, c1_kernels[p], rot_c1_kernels[p]);
   }
   for (int p=0;p<c1_kernel_count;p++) {
      c1_biases[p] -= learning_rate * delta_b1[p];
   }
   for (int q=0;q<c2_kernel_count;q++) {
      for (int p=0;p<c1_kernel_count;p++) {
         for (int x=0;x<kernel_size;x++) {
            for (int y=0;y<kernel_size;y++) {
               c2_kernels[q * c1_kernel_count + p][x * kernel_size +y] -= learning_rate * delta_c2_kernels[q * c1_kernel_count + p][x * kernel_size +y];
            }
         }
         if (debug == 1) {
            printf("c2 kernels %i %i after update\n", q, p);
            print_2D_array(c2_kernels[q * c1_kernel_count + p], kernel_size, kernel_size);
         }
         rotate_180(kernel_size, c2_kernels[q * c1_kernel_count + p], rot_c2_kernels[q * c1_kernel_count + p]);
      }
   }
   for (int q=0;q<c2_kernel_count;q++) {
      c2_biases[q] -= learning_rate * delta_b2[q];
   }
   for (int x=0;x<fc_output_size;x++) {
      for (int y=0;y<fc_input_size;y++) {
          fc_weights[x * fc_input_size + y] -= learning_rate * delta_W[x * fc_input_size + y];
      }
   }
   for (int r=0;r<fc_output_size;r++) {
      fc_biases[r] -= learning_rate * delta_fc_bias[r];
   }
   if (debug == 1) {
      printf("c1_biases\n");
      print_2D_array(c1_biases, c1_kernel_count, 1);
      printf("c2_biases\n");
      print_2D_array(c2_biases, c2_kernel_count, 1);
      printf("W after update\n");
      print_2D_array(fc_weights, fc_output_size, fc_input_size);
      printf("fc_biases\n");
      print_2D_array(fc_biases, fc_output_size, 1);
   }

}

int get_last_activated_output( SV *R ) {

    // Transfer results from host to perl
    AV *av, *av2;
    float *pd;
    size_t i,j,RH,RW, asz;

    RW = batch_size;
    RH = fc_output_size;

    if( is_array_ref(R, &asz) ){
            av = (AV *)SvRV(R);
            if( asz > 0 ){
               av_clear(av);
            }
    } else if( SvROK(R) ){
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(SvRV(R), (SV *)av);
    } else {
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(R, (SV *)av);
    }

    av = (AV *)SvRV(R);
    for(i=0;i<RH;i++){ // for each row
        av2 = newAV(); // make a new array for each row
        av_extend(av2, RH); // extend it to hold #cols items (RW)
        // LeoNerd's suggestion
        av_push(av, newRV_noinc((SV *)av2)); // insert it into the top Array
        for(j=0;j<RW;j++){ // for the cols of that row
            av_store(av2, j, newSVnv(y[i * batch_size + j]));
            pd++;
        }
    }

    return 0;
}

EOCODE


};


use Inline CPP => Config =>
        BUILD_NOISY => 0,
        force_build => 0,
        clean_after_build => 0,
        warnings => 0,
        INC => "-I" . abs_path("./inc") . " -I" . abs_path("./amd_kernel"),
        LIBS => "-L" . abs_path("./amd_kernel") . " -lKernels"
;

use Inline CPP => $code;



sub new {
   my $class = shift;
   my $self = {};
   my %args = @_;
   if (defined($args{debug}) and $args{debug} == 1) {
      $self->{debug} = 1;
      set_debug();
   } else {
      $self->{debug} = 0;
      unset_debug();
   }
   my $kernel_size = $args{kernel_size};
   if (!defined($kernel_size) or $kernel_size !~ /^\d+$/ or $kernel_size > $kernel_limit) {
      die "Invalid kernel_size parameter, must be integer <= $kernel_limit";
   }
   my $c1_kernels = $args{c1_kernels};  
   if (!defined($c1_kernels) or $c1_kernels !~ /^\d+$/ or $c1_kernels > $c1_kernels_limit) {
      die "Invalid c1_kernels parameter, must be integer <= $c1_kernels_limit";
   }
   my $input_dimensions = $args{input_dimensions};  
   if (!defined($input_dimensions) or ref($input_dimensions) ne "ARRAY" or $input_dimensions->[0] !~ /^\d+$/ or $input_dimensions->[1] !~ /^\d+$/) {
      die "Invalid input_dimensions parameter, must be array refs of 2 integers, e.g. [x, y]";
   }
   # so now we know the size and number of kernels in the first Convolution layer 
   my $kernels;
   my $c2_kernels = $c1_kernels * 2;
   init_c1_kernels($c1_kernels);
   init_c2_kernels($c2_kernels, $c1_kernels);
   my $c1_dist = sqrt($c1_kernels/((1 + $c1_kernels) * $kernel_size**2));
   foreach my $p (0 .. $c1_kernels - 1) {
      foreach my $x (0 .. $kernel_size - 1) {
         foreach my $y (0 .. $kernel_size - 1) {
            $kernels->[0][$p][$x][$y] = random_uniform(1, -1 * $c1_dist , $c1_dist);
         }
      }
      add_c1_kernel($p, $kernels->[0][$p]);
   }

# calculate sizes for various layers
   $self->{c1_kernels} = $c1_kernels;
   $self->{c2_kernels} = $c2_kernels; # are these known as "filters" in pytorch?
   $self->{kernel_size} = $kernel_size;
   $self->{range_upper} = ($self->{kernel_size} % 2 == 0)?($self->{kernel_size} / 2):( ($self->{kernel_size} - 1)/2 );
   $self->{range_lower} = -1 * $self->{range_upper};
   my $fc_input_size_side = $input_dimensions->[0]; 
   if (!$self->{padding}) {
      $fc_input_size_side -= $self->{range_upper} * 2; # after c1, the data will be kernel_size - 1 smaller each side
   }
   $self->{c1_size} = $fc_input_size_side; # needed for backprop
   $fc_input_size_side /= 2; # after first pooling size will be halved
   # this is the s1 size, we will need it later in backprop 
   $self->{s1_size} = $fc_input_size_side;
   if (!$self->{padding}) {
      $fc_input_size_side -= $self->{range_upper} * 2; # after c2, the data will be kernel_size - 1 smaller each side
   }
   # we will need c2_size on the backprop run
   $self->{c2_size} = $fc_input_size_side;
   $fc_input_size_side /= 2; # after second pooling size will be halved
   $self->{s2_size} = $fc_input_size_side;
   $self->{fc_input_size} = $fc_input_size_side * $fc_input_size_side * $self->{c2_kernels}; # output of second pooling is a square, and we have c2_kernels of them
   $self->{fc_output_size} = 10;

   init_params($self->{kernel_size}, 
               $self->{c1_kernels}, $self->{c2_kernels}, 
               $input_dimensions->[0], # input should be a square
               $self->{c1_size},
               $self->{s1_size},
               $self->{s2_size},
               $self->{c2_size},
               $self->{fc_input_size},
               $self->{fc_output_size}
              ); 
   my $biases;
   foreach my $p (0 .. $c1_kernels - 1) {
      $biases->[0][$p] = 0;
   }    
   set_c1_biases($biases->[0]);
   foreach my $q (0 .. $c2_kernels - 1) {
      $biases->[1][$q] = 0;
   }    
   set_c2_biases($biases->[1]);
   my $c2_dist = sqrt($c1_kernels/(($c1_kernels + $c2_kernels) * $kernel_size**2));
   foreach my $q (0 .. $c2_kernels - 1) {
      foreach my $p (0 .. $c1_kernels - 1) {
         foreach my $x (0 .. $kernel_size - 1) {
            foreach my $y (0 .. $kernel_size - 1) {
               $kernels->[1][$q][$p][$x][$y] = random_uniform(1, -1 * $c2_dist , $c2_dist);
            }
         }
         add_c2_kernel($q, $p, $c1_kernels, $kernels->[1][$q][$p]);
      }
   }

   $self->{biases} = $biases;
   $self->{input_dimensions} = $input_dimensions;
   $self->{kernels} = $kernels;
   $self->{c1} = []; # (input_dimension[0] - 2 * range_upper)^2, if no padding
   $self->{c2} = []; # (c1 - 2 * range_upper)^2, if no padding
   $self->{padding} = 0; # for future use, at the moment, no padding is applied, so C1 is kernel_size smaller than the input

# FC weights.  For demo purposes the output layer has 10 values.
   foreach my $b (0 .. $self->{fc_output_size} - 1) {
      $biases->[2][$b] = 0;
   }    
   set_fc_biases($biases->[2]);

   my $w_dist = sqrt($c1_kernels/($self->{fc_input_size} + $self->{fc_output_size}));
   $self->{W} = [];
   foreach my $row ( 0 .. $self->{fc_output_size} - 1) {
      foreach my $col ( 0 .. $self->{fc_input_size} - 1) {
          $self->{W}[$row][$col] = random_uniform(1, -1 * $w_dist, $w_dist);
      }
   }
   set_fc_weights($self->{W});
   if (defined($args{save_initial_weights})) {
      my $data = {
         weights => $self->{W},
         kernels => $self->{kernels}
      };
      open FILE, ">", "initial_weights.json";
      print FILE to_json($data);
      close FILE;
   }

   return bless $self, $class;

}

sub load_label {
   my $self = shift;
   $self->{label} = shift;
   set_label($self->{label});
}

sub forward {
   my $self = shift;
   my $input = shift;
   $self->{input} = $input; # needed for backprop
   load_input($self->{input});
   c1_conv(); # the c version of the 2d conv from input to c1 using the c1 kernels
   c1_pooling();
   c2_conv();
   c2_pooling();
   vectorise();
   final_connected_layer();
}
       
sub loss {
   my $self = shift;
   return calculate_loss();
}

sub results {
   my $self = shift;
   my $output = shift;
   return get_last_activated_output($output);
}

sub backward {
   my $self = shift;

   fc_delta();

   fc_bias_delta();

   w_delta();

   f_delta();

   p2_delta();

   c2_delta();

   k2_delta();

   b2_delta();
   
   p1_delta();

   c1_delta();

   c1_delta_sigmoid();
   
   k1_delta();

   b1_delta();

}

sub update_weights_and_biases {
   my $self = shift;
   my %params = @_;
   $params{batch_size} ||= 10;
   $params{learning_rate} ||= 3;
   $params{decay} ||= 1;
   update_params( $params{learning_rate}, $params{decay} );
}
1;
