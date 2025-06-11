/* Machine-generated using Migen */
module FIFO_TS(
	input link_ready,
	input reset,
	input re,
	input we,
	output fifo_full,
	output fifo_empty,
	input mclk,
	input oeiclk,
	output reg [39:0] tx_data,
	input [39:0] data_in
);

wire wr_clk;
wire wr_rst;
wire tx_clk;
wire tx_rst;
wire asyncfifo_we;
wire asyncfifo_writable;
wire asyncfifo_re;
wire asyncfifo_readable;
wire [39:0] asyncfifo_din;
wire [39:0] asyncfifo_dout;
wire graycounter0_ce;
(* no_retiming = "true" *) reg [8:0] graycounter0_q = 9'd0;
wire [8:0] graycounter0_q_next;
reg [8:0] graycounter0_q_binary = 9'd0;
reg [8:0] graycounter0_q_next_binary;
wire graycounter1_ce;
(* no_retiming = "true" *) reg [8:0] graycounter1_q = 9'd0;
wire [8:0] graycounter1_q_next;
reg [8:0] graycounter1_q_binary = 9'd0;
reg [8:0] graycounter1_q_next_binary;
wire [8:0] produce_rdomain;
wire [8:0] consume_wdomain;
wire [7:0] wrport_adr;
wire [39:0] wrport_dat_r;
wire wrport_we;
wire [39:0] wrport_dat_w;
wire [7:0] rdport_adr;
wire [39:0] rdport_dat_r;
(* no_retiming = "true" *) reg [8:0] multiregimpl0_regs0 = 9'd0;
(* no_retiming = "true" *) reg [8:0] multiregimpl0_regs1 = 9'd0;
(* no_retiming = "true" *) reg [8:0] multiregimpl1_regs0 = 9'd0;
(* no_retiming = "true" *) reg [8:0] multiregimpl1_regs1 = 9'd0;

// synthesis translate_off
reg dummy_s;
initial dummy_s <= 1'd0;
// synthesis translate_on

assign wr_rst = reset;
assign wr_clk = mclk;
assign tx_rst = reset;
assign tx_clk = oeiclk;
assign asyncfifo_din = data_in;
assign asyncfifo_we = we;
assign asyncfifo_re = re;
assign fifo_empty = (~asyncfifo_readable);
assign fifo_full = (~asyncfifo_writable);

// synthesis translate_off
reg dummy_d;
// synthesis translate_on
always @(*) begin
	tx_data <= 39'd0;
	if ((link_ready & asyncfifo_readable)) begin
		tx_data <= asyncfifo_dout[39:0];
	end
// synthesis translate_off
	dummy_d <= dummy_s;
// synthesis translate_on
end
assign graycounter0_ce = (asyncfifo_writable & asyncfifo_we);
assign graycounter1_ce = (asyncfifo_readable & asyncfifo_re);
assign asyncfifo_writable = (((graycounter0_q[8] == consume_wdomain[8]) | (graycounter0_q[7] == consume_wdomain[7])) | (graycounter0_q[6:0] != consume_wdomain[6:0]));
assign asyncfifo_readable = (graycounter1_q != produce_rdomain);
assign wrport_adr = graycounter0_q_binary[7:0];
assign wrport_dat_w = asyncfifo_din;
assign wrport_we = graycounter0_ce;
assign rdport_adr = graycounter1_q_next_binary[7:0];
assign asyncfifo_dout = rdport_dat_r;

// synthesis translate_off
reg dummy_d_1;
// synthesis translate_on
always @(*) begin
	graycounter0_q_next_binary <= 9'd0;
	if (graycounter0_ce) begin
		graycounter0_q_next_binary <= (graycounter0_q_binary + 1'd1);
	end else begin
		graycounter0_q_next_binary <= graycounter0_q_binary;
	end
// synthesis translate_off
	dummy_d_1 <= dummy_s;
// synthesis translate_on
end
assign graycounter0_q_next = (graycounter0_q_next_binary ^ graycounter0_q_next_binary[8:1]);

// synthesis translate_off
reg dummy_d_2;
// synthesis translate_on
always @(*) begin
	graycounter1_q_next_binary <= 9'd0;
	if (graycounter1_ce) begin
		graycounter1_q_next_binary <= (graycounter1_q_binary + 1'd1);
	end else begin
		graycounter1_q_next_binary <= graycounter1_q_binary;
	end
// synthesis translate_off
	dummy_d_2 <= dummy_s;
// synthesis translate_on
end
assign graycounter1_q_next = (graycounter1_q_next_binary ^ graycounter1_q_next_binary[8:1]);
assign produce_rdomain = multiregimpl0_regs1;
assign consume_wdomain = multiregimpl1_regs1;

always @(posedge tx_clk) begin
	graycounter1_q_binary <= graycounter1_q_next_binary;
	graycounter1_q <= graycounter1_q_next;
	if (tx_rst) begin
		graycounter1_q <= 9'd0;
		graycounter1_q_binary <= 9'd0;
	end
	multiregimpl0_regs0 <= graycounter0_q;
	multiregimpl0_regs1 <= multiregimpl0_regs0;
end

always @(posedge wr_clk) begin
	graycounter0_q_binary <= graycounter0_q_next_binary;
	graycounter0_q <= graycounter0_q_next;
	if (wr_rst) begin
		graycounter0_q <= 9'd0;
		graycounter0_q_binary <= 9'd0;
	end
	multiregimpl1_regs0 <= graycounter1_q;
	multiregimpl1_regs1 <= multiregimpl1_regs0;
end

reg [39:0] storage[0:255];
reg [7:0] memadr;
reg [7:0] memadr_1;
always @(posedge wr_clk) begin
	if (wrport_we)
		storage[wrport_adr] <= wrport_dat_w;
	memadr <= wrport_adr;
end

always @(posedge tx_clk) begin
	memadr_1 <= rdport_adr;
end

assign wrport_dat_r = storage[memadr];
assign rdport_dat_r = storage[memadr_1];

endmodule
