extern crate proc_macro;
use proc_macro::{Span, TokenStream};
use quote::ToTokens;
use syn::{
    parse::{Parse, ParseStream},
    spanned::Spanned,
    ExprUnsafe,
};

struct ExprGpuAsync {
    pub stmts: Vec<syn::Stmt>,
}
impl Parse for ExprGpuAsync {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(ExprGpuAsync {
            stmts: syn::Block::parse_within(input)?,
        })
    }
}

/// Transform xxxxx.await into
/// ```rs
/// {
///     let future = xxxx;
///    hook::new(&future).await;
///    future
/// }.await;
/// ```
fn transform_await_expr(expr: &syn::ExprAwait) -> syn::Expr {
    let base = expr.base.clone();
    syn::parse_quote! {
        {
            let future = #base;
            async_ash::future::GPUFutureHook::new(&future).await;
            future.await
        }
    }
}

#[proc_macro]
pub fn gpu(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as ExprGpuAsync);

    let op_collector_block = syn::Block {
        brace_token: Default::default(),
        stmts: input
            .stmts
            .iter()
            .map(|stmt| {
                use syn::{Block, Expr, Pat, Stmt};
                fn transform_block(block: &Block) -> Block {
                    Block {
                        brace_token: block.brace_token.clone(),
                        stmts: block.stmts.iter().map(transform_stmt).collect(),
                    }
                }
                fn transform_pattern(pat: &Pat) -> Pat {
                    match pat {
                        Pat::Box(pat) => Pat::Box(syn::PatBox {
                            pat: Box::new(transform_pattern(&pat.pat)),
                            ..pat.clone()
                        }),
                        Pat::Ident(ident) => Pat::Ident(syn::PatIdent {
                            subpat: ident.subpat.as_ref().map(|(at, subpat)| {
                                (at.clone(), Box::new(transform_pattern(subpat)))
                            }),
                            ..ident.clone()
                        }),
                        Pat::Lit(lit) => Pat::Lit(syn::PatLit {
                            expr: Box::new(transform_expr(&lit.expr)),
                            ..lit.clone()
                        }),
                        Pat::Or(clause) => Pat::Or(syn::PatOr {
                            cases: clause.cases.iter().map(transform_pattern).collect(),
                            ..clause.clone()
                        }),
                        Pat::Range(range) => Pat::Range(syn::PatRange {
                            lo: Box::new(transform_expr(&range.lo)),
                            hi: Box::new(transform_expr(&range.hi)),
                            ..range.clone()
                        }),
                        Pat::Reference(r) => Pat::Reference(syn::PatReference {
                            pat: Box::new(transform_pattern(&r.pat)),
                            ..r.clone()
                        }),
                        Pat::Slice(slice) => Pat::Slice(syn::PatSlice {
                            elems: slice.elems.iter().map(transform_pattern).collect(),
                            ..slice.clone()
                        }),
                        Pat::Struct(s) => Pat::Struct(syn::PatStruct {
                            fields: s
                                .fields
                                .iter()
                                .map(|f| syn::FieldPat {
                                    pat: Box::new(transform_pattern(&f.pat)),
                                    ..f.clone()
                                })
                                .collect(),
                            ..s.clone()
                        }),
                        Pat::Tuple(tuple) => Pat::Tuple(syn::PatTuple {
                            elems: tuple.elems.iter().map(transform_pattern).collect(),
                            ..tuple.clone()
                        }),
                        Pat::TupleStruct(tuple) => Pat::TupleStruct(syn::PatTupleStruct {
                            pat: syn::PatTuple {
                                elems: tuple.pat.elems.iter().map(transform_pattern).collect(),
                                ..tuple.pat.clone()
                            },
                            ..tuple.clone()
                        }),
                        Pat::Type(ty) => Pat::Type(syn::PatType {
                            pat: Box::new(transform_pattern(&ty.pat)),
                            ..ty.clone()
                        }),
                        _ => pat.clone(),
                    }
                }
                fn transform_expr(expr: &Expr) -> Expr {
                    match expr {
                        Expr::Array(arr) => Expr::Array(syn::ExprArray {
                            elems: arr.elems.iter().map(transform_expr).collect(),
                            ..arr.clone()
                        }),
                        Expr::Assign(assign) => Expr::Assign(syn::ExprAssign {
                            left: Box::new(transform_expr(&*assign.left)),
                            right: Box::new(transform_expr(&*assign.right)),
                            ..assign.clone()
                        }),
                        Expr::AssignOp(assign) => Expr::AssignOp(syn::ExprAssignOp {
                            left: Box::new(transform_expr(&*assign.left)),
                            right: Box::new(transform_expr(&*assign.right)),
                            ..assign.clone()
                        }),
                        Expr::Binary(binary) => Expr::Binary(syn::ExprBinary {
                            left: Box::new(transform_expr(&*binary.left)),
                            right: Box::new(transform_expr(&*binary.right)),
                            ..binary.clone()
                        }),
                        Expr::Block(block) => Expr::Block(syn::ExprBlock {
                            block: transform_block(&block.block),
                            ..block.clone()
                        }),
                        Expr::Call(call) => Expr::Call(syn::ExprCall {
                            func: Box::new(transform_expr(&*call.func)),
                            args: call.args.iter().map(transform_expr).collect(),
                            ..call.clone()
                        }),
                        Expr::Cast(cast) => Expr::Cast(syn::ExprCast {
                            expr: Box::new(transform_expr(&*cast.expr)),
                            ..cast.clone()
                        }),
                        Expr::Field(field) => Expr::Field(syn::ExprField {
                            base: Box::new(transform_expr(&*field.base)),
                            ..field.clone()
                        }),
                        Expr::ForLoop(l) => Expr::ForLoop(syn::ExprForLoop {
                            pat: transform_pattern(&l.pat),
                            expr: Box::new(transform_expr(&*l.expr)),
                            body: transform_block(&l.body),
                            ..l.clone()
                        }),
                        Expr::Group(group) => Expr::Group(syn::ExprGroup {
                            expr: Box::new(transform_expr(&group.expr)),
                            ..group.clone()
                        }),
                        Expr::If(if_stmt) => Expr::If(syn::ExprIf {
                            cond: Box::new(transform_expr(&if_stmt.cond)),
                            then_branch: transform_block(&if_stmt.then_branch),
                            else_branch: if_stmt.else_branch.as_ref().map(|(else_token, expr)| {
                                (else_token.clone(), Box::new(transform_expr(&expr)))
                            }),
                            ..if_stmt.clone()
                        }),
                        Expr::Index(index_expr) => Expr::Index(syn::ExprIndex {
                            expr: Box::new(transform_expr(&*index_expr.expr)),
                            index: Box::new(transform_expr(&*index_expr.index)),
                            ..index_expr.clone()
                        }),
                        Expr::Let(l) => Expr::Let(syn::ExprLet {
                            pat: transform_pattern(&l.pat),
                            expr: Box::new(transform_expr(&*l.expr)),
                            ..l.clone()
                        }),
                        Expr::Loop(loop_stmt) => Expr::Loop(syn::ExprLoop {
                            body: transform_block(&loop_stmt.body),
                            ..loop_stmt.clone()
                        }),
                        Expr::Match(match_expr) => Expr::Match(syn::ExprMatch {
                            expr: Box::new(transform_expr(&*match_expr.expr)),
                            arms: match_expr
                                .arms
                                .iter()
                                .map(|arm| syn::Arm {
                                    pat: transform_pattern(&arm.pat),
                                    guard: arm.guard.as_ref().map(|(guard_token, expr)| {
                                        (guard_token.clone(), Box::new(transform_expr(&expr)))
                                    }),
                                    body: Box::new(transform_expr(&arm.body)),
                                    ..arm.clone()
                                })
                                .collect(),
                            ..match_expr.clone()
                        }),
                        Expr::MethodCall(call) => Expr::MethodCall(syn::ExprMethodCall {
                            receiver: Box::new(transform_expr(&call.receiver)),
                            args: call.args.iter().map(transform_expr).collect(),
                            ..call.clone()
                        }),
                        Expr::Paren(paren) => Expr::Paren(syn::ExprParen {
                            expr: Box::new(transform_expr(&paren.expr)),
                            ..paren.clone()
                        }),
                        Expr::Range(range) => Expr::Range(syn::ExprRange {
                            from: range.from.as_ref().map(|f| Box::new(transform_expr(&f))),
                            to: range.to.as_ref().map(|t| Box::new(transform_expr(&t))),
                            ..range.clone()
                        }),
                        Expr::Reference(reference) => Expr::Reference(syn::ExprReference {
                            expr: Box::new(transform_expr(&reference.expr)),
                            ..reference.clone()
                        }),
                        Expr::Repeat(repeat) => Expr::Repeat(syn::ExprRepeat {
                            expr: Box::new(transform_expr(&repeat.expr)),
                            len: Box::new(transform_expr(&repeat.len)),
                            ..repeat.clone()
                        }),
                        Expr::Return(ret) => Expr::Return(syn::ExprReturn {
                            expr: ret.expr.as_ref().map(|e| Box::new(transform_expr(&e))),
                            ..ret.clone()
                        }),
                        Expr::Struct(s) => Expr::Struct(syn::ExprStruct {
                            fields: s
                                .fields
                                .iter()
                                .map(|f| syn::FieldValue {
                                    member: f.member.clone(),
                                    expr: transform_expr(&f.expr),
                                    ..f.clone()
                                })
                                .collect(),
                            ..s.clone()
                        }),
                        Expr::Try(s) => Expr::Try(syn::ExprTry {
                            expr: Box::new(transform_expr(&s.expr)),
                            ..s.clone()
                        }),
                        Expr::Tuple(tuple) => Expr::Tuple(syn::ExprTuple {
                            elems: tuple.elems.iter().map(transform_expr).collect(),
                            ..tuple.clone()
                        }),
                        Expr::Type(type_expr) => Expr::Type(syn::ExprType {
                            expr: Box::new(transform_expr(&type_expr.expr)),
                            ..type_expr.clone()
                        }),
                        Expr::Unary(unary) => Expr::Unary(syn::ExprUnary {
                            expr: Box::new(transform_expr(&unary.expr)),
                            ..unary.clone()
                        }),
                        Expr::Unsafe(unsafe_stmt) => Expr::Unsafe(ExprUnsafe {
                            block: transform_block(&unsafe_stmt.block),
                            ..unsafe_stmt.clone()
                        }),
                        Expr::While(while_stmt) => Expr::While(syn::ExprWhile {
                            cond: Box::new(transform_expr(&while_stmt.cond)),
                            body: transform_block(&while_stmt.body),
                            ..while_stmt.clone()
                        }),
                        Expr::Await(await_expr) => transform_await_expr(await_expr),
                        _ => expr.clone(),
                    }
                }
                fn transform_stmt(stmt: &Stmt) -> Stmt {
                    match stmt {
                        Stmt::Local(local) => Stmt::Local(syn::Local {
                            pat: transform_pattern(&local.pat),
                            init: local.init.as_ref().map(|(eq_token, expr)| {
                                (eq_token.clone(), Box::new(transform_expr(&expr)))
                            }),
                            ..local.clone()
                        }),
                        Stmt::Expr(expr) => Stmt::Expr(transform_expr(expr)),
                        Stmt::Semi(expr, semi) => Stmt::Semi(transform_expr(expr), semi.clone()),
                        _ => stmt.clone(),
                    }
                }
                transform_stmt(stmt)
            })
            .collect(),
    };

    let async_block = syn::ExprAsync {
        attrs: Vec::new(),
        async_token: Default::default(),
        capture: None,
        block: op_collector_block,
    };

    quote::quote! {
        async_ash::future::GPUFutureBlock::new(#async_block)
    }
    .into()
}
