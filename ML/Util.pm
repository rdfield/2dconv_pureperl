use Modern::Perl;
package ML::Util;
use Exporter 'import';
our @EXPORT_OK = qw(print_2d_array transpose add_2_arrays diagonal_matrix matmul print_1d_array rotate_matrix_180 conv2d);

sub print_2d_array {
   my ($title,$d) = @_;
   say $title;
   say "rows = " . scalar(@$d) . " columns = " . scalar(@{$d->[0]});
   foreach my $row (@$d) {
      say join("\t", map { sprintf("%+.5f",$_) } @{$row});
   }
   say "";
}

sub print_1d_array {
   my ($title,$d) = @_;
   say $title;
   say join("\t", map { sprintf("%+.5f",$_) } @{$d});
}

sub transpose {
   my $M = shift;   
   my $O = []; 
   foreach my $i (0 .. $#$M) {
      foreach my $j (0 .. $#{$M->[0]}) {
         $O->[$j][$i] = $M->[$i][$j];
      }
   }
   return $O;   
}

sub add_2_arrays { # same size assumed
   my ($M1, $M2) = @_;
   my $O = [];
   foreach my $i (0 .. $#$M1) {
      foreach my $j (0 .. $#{$M1->[0]}) {
         $O->[$i][$j] = $M1->[$i][$j] + $M2->[$i][$j];
      } 
   }
   return $O;
} 

sub diagonal_matrix {
   my ($value, $max_index) = @_;
   my $O = [];
   foreach my $i (0 .. $max_index) {
      foreach my $j (0 .. $max_index) {
         if ($i == $j) {
            $O->[$i][$j] = $value;
         } else {
            $O->[$i][$j] = 0;  
         }
      }
   }
   return $O;
}

sub matmul {
   # matrix multiplication
   my ($il, $ol) = @_;
   my $ar = $#$il;
   my $ac = $#{$il->[0]};
   my $br = $#$ol;
   my $bc = $#{$ol->[0]};
   do {
      say "matmul: ar = $ar, ac = $ac, br = $br, bc = $bc";
      die "matmul: input layer output not the same length as input of output layer, called by " . join(",",caller())
   } unless $ac == $br;
   my $c = [];
   foreach my $i ( 0 .. $ar) {
      foreach my $j ( 0  .. $bc) {
         foreach my $k ( 0 .. $ac) {
            $c->[$i][$j] += $il->[$i][$k] * $ol->[$k][$j];
#            say "c->[$i][$j] = " . $c->[$i][$j];
         }
      }
   }
   #say "matmul result = " . Dumper($c);
   return $c;
}

sub rotate_matrix_180 {
   my $A = shift;
   my $A_rot = [];

   my $A_x = $#{$A};
   my $A_y = $#{$A->[0]};
   foreach my $i (0 .. $A_x) {
      foreach my $j (0 .. $A_y) {
         $A_rot->[$A_x - $i][$A_y - $j] = $A->[$i][$j];
      }
   }
   return $A_rot;
}

sub conv2d {
  my ($in, $filter, $operation) = @_;
  $operation = "padded" unless $operation eq "expand" or $operation eq "reduce";
  # operation:
  #    expand: the convolution starts at the $in(0, 0) cell, which means the output is $filter/2 cells bigger than $in in each axis
  #    padded: the convolution starts at the $in($filter/2, $filter/2) cell, so that $output is the same size as $in
  #    reduce: the convolution starts at the $in($filter, $filter) cell, so that the $output is $filter/2 cells smaller than $in in each axis
  # in each case the kernel centre is set to the cell to be calculated, so for $in(i, j) the cells involved in the calc are $in(i - filter/2, j - filter/2) to $in(i + filter/2, j + filter/2)
  # and any negative indicies are return "0".
  # this function assumes that the $filter has NOT been rotated, so the first action is to rotate the filter
  # without rotating the filter it would be "correlation" rather than "convolution"
  my $rot_filter = rotate_matrix_180($filter);
  my $kernel_size = scalar(@$filter);
  my $input_size = scalar(@$in);
  my $kernel_offset = int( $kernel_size  / 2); 
  my $output = [];
  my $output_size = scalar(@$in); # start off with output being the same size as the input
  my $in_offset = 0;
  if ($operation eq "expand") {
     $output_size += $kernel_offset * 2;
     $in_offset = $kernel_offset * 2;
  }
  if ($operation eq "reduce") {
     $output_size -= $kernel_offset * 2 ;
     if ($kernel_size % 2 == 0) { # even sized kernel, add 1 to the output size
        $output_size++;
     }
  }
  if ($operation eq "padded") {
     $in_offset = $kernel_offset;
  }
  foreach my $i (0 .. $output_size - 1) {
     foreach my $j (0 .. $output_size - 1) {
        $output->[$i][$j] = 0;
        foreach my $u (0 .. $kernel_size - 1) {
           foreach my $v (0 .. $kernel_size - 1) {
              my $in_i = $i + $u - $in_offset;
              my $in_j = $j + $v - $in_offset;
              next if $in_i < 0 or $in_j < 0 or $in_i >= $input_size or $in_j >= $input_size;
              $output->[$i][$j] += $rot_filter->[$u][$v] * $in->[$in_i][$in_j];
           }
        }
     }
  }
  return $output;
}
         
1;
