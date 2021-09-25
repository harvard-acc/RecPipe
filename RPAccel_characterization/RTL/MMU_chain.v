`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Harvard University
// Jeff (Jun) Zhang
// jeffzhang@g.harvard.edu 

// Create Date:    18:34:05 04/22/2021  
// Design Name: 
// Module Name:    MMU 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
 

module MMU_chain #(parameter depth=128, bit_width=8, acc_width=24, size=128, mmu_block=16)
(
	clk,
	control,
	//data_arr,
	data_arr_alt,
	data_arr_sle_s,
	acc_arr_sle_d,
	wt_arr,
	acc_in,
	wt_arr_out,
	data_arr_out,
	acc_out
    );


localparam s_len = size / mmu_block;

input clk;
input  control;
//input  [bit_width*16-1:0] data_arr;
input  [bit_width*mmu_block*s_len-1:0] data_arr_alt;
input  [s_len-1:0] data_arr_sle_s;
input  [s_len-1:0] acc_arr_sle_d;
input  [bit_width*mmu_block-1:0] wt_arr;
input [acc_width*mmu_block*s_len-1 : 0] acc_in;

output reg [bit_width*mmu_block-1:0] wt_arr_out;
output reg [bit_width*mmu_block-1:0] data_arr_out;
output reg [acc_width*mmu_block*s_len-1 : 0] acc_out;
	


wire [acc_width*mmu_block-1:0] acc_out_temp [0: s_len-1];
reg [bit_width*mmu_block-1:0] data_arr_temp [0: s_len-1];

integer m,n;

always@* begin
	for(m=0;m<s_len;m=m+1)begin
	for(n=0;n<acc_width*mmu_block;n=n+1)begin
		acc_out[m*acc_width*mmu_block+n]=acc_out_temp[m][n];

	end
	end
	end

always@* begin
	for(m=0;m<s_len;m=m+1)begin
	for(n=0;n<bit_width*mmu_block;n=n+1)begin
		data_arr_temp[m][n] = data_arr_alt[m*bit_width*mmu_block+n];

	end
	end
	end

	reg [bit_width*mmu_block-1:0] temp_first_wt;
	wire [bit_width*mmu_block-1:0] temp_data [s_len:0];		//to connect MACs in the MAC_chain
	wire [bit_width*mmu_block-1:0] temp_data2 [s_len:0];
	wire [bit_width*mmu_block-1:0] temp_wt[s_len:0];
	wire [acc_width*mmu_block-1:0] acc_in2 [s_len-1:0];
	
	assign temp_wt[0]=temp_first_wt;

	always@(*)begin
	if(control) temp_first_wt=wt_arr; 
	else temp_first_wt= 0;
	end

	assign temp_data[0]=0;


    always@(*)begin
	//if(~control)
	data_arr_out = temp_data[s_len];
	wt_arr_out = temp_wt[s_len];
	//else
	//acc_out=0;
	end




	generate 
	genvar i;
	
    for(i=0;i<s_len;i=i+1)
		begin:mmu

		assign temp_data2[i] = (data_arr_sle_s[i])? data_arr_temp[i]: temp_data[i];
		assign acc_in2[i] = (acc_arr_sle_d[i])? 0: acc_in[i*acc_width*mmu_block+acc_width*mmu_block-1:i*acc_width*mmu_block];

        MMU #(.depth(mmu_block), .bit_width(bit_width),.acc_width(acc_width),.size(mmu_block)) mmu (
		.clk(clk),
		.control(control),
		.data_arr(temp_data2[i]),  // mux: temp_data[i], data_arr_alt
		.wt_arr(temp_wt[i]),
		//.acc_in(acc_in[i*acc_width*mmu_block+acc_width*mmu_block-1:i*acc_width*mmu_block]),
		.acc_in(acc_in2[i]),
		.data_arr_out(temp_data[i+1]),
		.wt_arr_out(temp_wt[i+1]),
		.acc_out(acc_out_temp[i])

		);
		
		end
		
	endgenerate






endmodule
