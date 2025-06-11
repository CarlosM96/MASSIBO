/* Machine-generated using Migen */
module FIFO16_2(
	input link_ready,
	input reset,
	input re,
	input we,
	output fifo_full,
	output fifo_empty,
	input mclk,
	input oeiclk,
	output reg [15:0] tx_data,
	output reg [14:0] wr_addr,
	input [15:0] data_in
);

wire wr_clk;
wire wr_rst;
wire tx_clk;
wire tx_rst;
wire asyncfifo_we;
wire asyncfifo_writable;
wire asyncfifo_re;
wire asyncfifo_readable;
wire [15:0] asyncfifo_din;
wire [15:0] asyncfifo_dout;
wire graycounter0_ce;
(* no_retiming = "true" *) reg [15:0] graycounter0_q = 16'd0;
wire [15:0] graycounter0_q_next;
reg [15:0] graycounter0_q_binary = 16'd0;
reg [15:0] graycounter0_q_next_binary;
wire graycounter1_ce;
(* no_retiming = "true" *) reg [15:0] graycounter1_q = 16'd0;
wire [15:0] graycounter1_q_next;
reg [15:0] graycounter1_q_binary = 16'd0;
reg [15:0] graycounter1_q_next_binary;
wire [15:0] produce_rdomain;
wire [15:0] consume_wdomain;
wire [14:0] wrport_adr;
wire [15:0] wrport_dat_r;
wire wrport_we;
wire [15:0] wrport_dat_w;
wire [14:0] rdport_adr;
wire [15:0] rdport_dat_r;
(* no_retiming = "true" *) reg [15:0] multiregimpl0_regs0 = 16'd0;
(* no_retiming = "true" *) reg [15:0] multiregimpl0_regs1 = 16'd0;
(* no_retiming = "true" *) reg [15:0] multiregimpl1_regs0 = 16'd0;
(* no_retiming = "true" *) reg [15:0] multiregimpl1_regs1 = 16'd0;

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
	tx_data <= 16'd0;
	if ((link_ready & asyncfifo_readable)) begin
		tx_data <= asyncfifo_dout[15:0];
	end
// synthesis translate_off
	dummy_d <= dummy_s;
// synthesis translate_on
end
assign graycounter0_ce = (asyncfifo_writable & asyncfifo_we);
assign graycounter1_ce = (asyncfifo_readable & asyncfifo_re);
assign asyncfifo_writable = (((graycounter0_q[15] == consume_wdomain[15]) | (graycounter0_q[14] == consume_wdomain[14])) | (graycounter0_q[13:0] != consume_wdomain[13:0]));
assign asyncfifo_readable = (graycounter1_q != produce_rdomain);
assign wrport_adr = graycounter0_q_binary[14:0];
assign wrport_dat_w = asyncfifo_din;
assign wrport_we = graycounter0_ce;
assign rdport_adr = graycounter1_q_next_binary[14:0];
assign asyncfifo_dout = rdport_dat_r;

// synthesis translate_off
reg dummy_d_1;
// synthesis translate_on
always @(*) begin
	graycounter0_q_next_binary <= 16'd0;
	if (graycounter0_ce) begin
		graycounter0_q_next_binary <= (graycounter0_q_binary + 1'd1);
	end else begin
		graycounter0_q_next_binary <= graycounter0_q_binary;
	end
// synthesis translate_off
	dummy_d_1 <= dummy_s;
// synthesis translate_on
end
assign graycounter0_q_next = (graycounter0_q_next_binary ^ graycounter0_q_next_binary[15:1]);

// synthesis translate_off
reg dummy_d_2;
// synthesis translate_on
always @(*) begin
	graycounter1_q_next_binary <= 16'd0;
	if (graycounter1_ce) begin
		graycounter1_q_next_binary <= (graycounter1_q_binary + 1'd1);
	end else begin
		graycounter1_q_next_binary <= graycounter1_q_binary;
	end
// synthesis translate_off
	dummy_d_2 <= dummy_s;
// synthesis translate_on
end
assign graycounter1_q_next = (graycounter1_q_next_binary ^ graycounter1_q_next_binary[15:1]);
assign produce_rdomain = multiregimpl0_regs1;
assign consume_wdomain = multiregimpl1_regs1;

always @(posedge tx_clk) begin
	graycounter1_q_binary <= graycounter1_q_next_binary;
	graycounter1_q <= graycounter1_q_next;
	if (tx_rst) begin
		graycounter1_q <= 16'd0;
		graycounter1_q_binary <= 16'd0;
	end
	multiregimpl0_regs0 <= graycounter0_q;
	multiregimpl0_regs1 <= multiregimpl0_regs0;
end

always @(posedge wr_clk) begin
	graycounter0_q_binary <= graycounter0_q_next_binary;
	graycounter0_q <= graycounter0_q_next;
	if (wr_rst) begin
		graycounter0_q <= 16'd0;
		graycounter0_q_binary <= 16'd0;
	end
	multiregimpl1_regs0 <= graycounter1_q;
	multiregimpl1_regs1 <= multiregimpl1_regs0;
end

reg [15:0] storage[0:32767];
reg [14:0] memadr;
reg [14:0] memadr_1;
always @(posedge wr_clk) begin
	if (wrport_we)
		storage[wrport_adr] <= wrport_dat_w;
        memadr <= wrport_adr;
        wr_addr <= wrport_adr;
end

always @(posedge tx_clk) begin
	memadr_1 <= rdport_adr;
end

assign wrport_dat_r = storage[memadr];
assign rdport_dat_r = storage[memadr_1];

endmodule
